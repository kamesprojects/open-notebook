"""API tests for the embedding rebuild router.

The endpoints are called through the FastAPI TestClient. repo_query,
CommandService.submit_command_job and get_command_status are replaced by
test doubles, so the tests check only the router logic: the item count
estimate, the submitted command, the status response and error handling.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from open_notebook.exceptions import NotFoundError

MODULE = "api.routers.embedding_rebuild"
REBUILD_URL = "/api/embeddings/rebuild"
STATUS_URL = "/api/embeddings/rebuild/command:1/status"


@pytest.fixture
def client():
    from api.main import app

    return TestClient(app)


def mock_submit(command_id="command:1"):
    return patch(
        f"{MODULE}.CommandService.submit_command_job",
        AsyncMock(return_value=command_id),
    )


def mock_status(status):
    return patch(f"{MODULE}.get_command_status", AsyncMock(return_value=status))


class TestStartRebuild:
    def test_existing_mode_counts_and_submits(self, client):
        # sources, notes, insights; the count comes back as a dict, a bare
        # number, or nothing at all
        query = AsyncMock(side_effect=[[{"count": 2}], [3], []])
        submit = AsyncMock(return_value="command:1")

        with (
            patch(f"{MODULE}.repo_query", query),
            patch(f"{MODULE}.CommandService.submit_command_job", submit),
        ):
            response = client.post(REBUILD_URL, json={"mode": "existing"})

        assert response.status_code == 200
        assert response.json()["command_id"] == "command:1"
        assert response.json()["total_items"] == 5
        assert query.await_count == 3
        submit.assert_awaited_once_with(
            "open_notebook",
            "rebuild_embeddings",
            {
                "mode": "existing",
                "include_sources": True,
                "include_notes": True,
                "include_insights": True,
            },
        )

    def test_all_mode_ignores_unexpected_values(self, client):
        query = AsyncMock(side_effect=[["not a number"], [{"count": 4}], [1]])

        with patch(f"{MODULE}.repo_query", query), mock_submit():
            response = client.post(REBUILD_URL, json={"mode": "all"})

        assert response.status_code == 200
        assert response.json()["total_items"] == 5

    @pytest.mark.parametrize("mode", ["existing", "all"])
    @pytest.mark.parametrize(
        "only", ["include_sources", "include_notes", "include_insights"]
    )
    def test_only_selected_types_are_counted(self, client, mode, only):
        flags = {
            "include_sources": False,
            "include_notes": False,
            "include_insights": False,
            only: True,
        }
        query = AsyncMock(return_value=[{"count": 7}])

        with patch(f"{MODULE}.repo_query", query), mock_submit():
            response = client.post(REBUILD_URL, json={"mode": mode, **flags})

        assert response.status_code == 200
        assert response.json()["total_items"] == 7
        assert query.await_count == 1

    @pytest.mark.parametrize(
        "error, status",
        [
            (HTTPException(status_code=418, detail="teapot"), 418),
            (NotFoundError("gone"), 404),
            (RuntimeError("boom"), 500),
        ],
    )
    def test_errors_map_to_status_codes(self, client, error, status):
        with patch(f"{MODULE}.repo_query", AsyncMock(side_effect=error)):
            response = client.post(REBUILD_URL, json={"mode": "all"})

        assert response.status_code == status


class TestGetRebuildStatus:
    def test_unknown_command_is_404(self, client):
        with mock_status(None):
            response = client.get(STATUS_URL)

        assert response.status_code == 404

    def test_running_command_with_progress_and_stats(self, client):
        status = SimpleNamespace(
            status="running",
            result={
                "total_items": 10,
                "jobs_submitted": 5,
                "sources_submitted": 2,
                "notes_submitted": 2,
                "insights_submitted": 1,
            },
            created="2026-01-01 12:00:00",
            updated="2026-01-01 12:05:00",
        )

        with mock_status(status):
            response = client.get(STATUS_URL)

        body = response.json()
        assert response.status_code == 200
        assert body["status"] == "running"
        assert body["progress"] == {"processed": 5, "total": 10, "percentage": 50.0}
        assert body["stats"] == {"sources": 2, "notes": 2, "insights": 1, "failed": 0}
        assert body["started_at"] == "2026-01-01 12:00:00"
        assert body["completed_at"] == "2026-01-01 12:05:00"
        assert body["error_message"] is None

    def test_failed_command_reports_error(self, client):
        status = SimpleNamespace(
            status="failed",
            result={"total_items": 0, "jobs_submitted": 0, "error_message": "oops"},
        )

        with mock_status(status):
            response = client.get(STATUS_URL)

        body = response.json()
        assert body["progress"]["percentage"] == 0
        assert body["error_message"] == "oops"
        assert body["started_at"] is None

    def test_queued_command_without_result(self, client):
        status = SimpleNamespace(status="queued", result=None, created=None)

        with mock_status(status):
            response = client.get(STATUS_URL)

        body = response.json()
        assert body["status"] == "queued"
        assert body["progress"] is None
        assert body["stats"] is None

    @pytest.mark.parametrize(
        "error, status",
        [(NotFoundError("gone"), 404), (RuntimeError("boom"), 500)],
    )
    def test_errors_map_to_status_codes(self, client, error, status):
        with patch(f"{MODULE}.get_command_status", AsyncMock(side_effect=error)):
            response = client.get(STATUS_URL)

        assert response.status_code == status
