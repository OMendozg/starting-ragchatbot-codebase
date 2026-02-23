"""
Tests for the FastAPI endpoints (/api/query, /api/courses).

Uses the `api_client` fixture from conftest.py, which provides a Starlette
TestClient backed by a minimal test app that mirrors app.py's endpoints
without mounting static files — avoiding import errors in test environments.

Tests are split by endpoint and cover:
- Successful request/response shapes
- Session handling (create vs. re-use)
- Delegation to the RAG system
- Validation errors (422) for malformed requests
- 500 responses when the RAG system raises an exception
"""
import pytest
from helpers import (
    SAMPLE_ANSWER,
    SAMPLE_SOURCES,
    SAMPLE_SESSION_ID,
    SAMPLE_COURSES,
)


# ---------------------------------------------------------------------------
# POST /api/query
# ---------------------------------------------------------------------------

class TestQueryEndpoint:

    def test_valid_query_returns_200(self, api_client):
        """A well-formed request body returns HTTP 200."""
        response = api_client.post("/api/query", json={"query": "What is Python?"})
        assert response.status_code == 200

    def test_response_contains_answer(self, api_client):
        """Response body includes the answer string from the RAG system."""
        response = api_client.post("/api/query", json={"query": "What is Python?"})
        assert response.json()["answer"] == SAMPLE_ANSWER

    def test_response_contains_sources(self, api_client):
        """Response body includes the sources list from the RAG system."""
        response = api_client.post("/api/query", json={"query": "What is Python?"})
        assert response.json()["sources"] == SAMPLE_SOURCES

    def test_response_contains_session_id(self, api_client):
        """Response body includes a non-empty session_id."""
        response = api_client.post("/api/query", json={"query": "What is Python?"})
        data = response.json()
        assert "session_id" in data
        assert data["session_id"]

    def test_no_session_id_creates_new_session(self, api_client, mock_rag_system):
        """When session_id is omitted, session_manager.create_session() is called."""
        api_client.post("/api/query", json={"query": "What is Python?"})
        mock_rag_system.session_manager.create_session.assert_called_once()

    def test_no_session_id_response_uses_created_session(self, api_client):
        """When session_id is omitted, the response session_id is the newly created one."""
        response = api_client.post("/api/query", json={"query": "What is Python?"})
        assert response.json()["session_id"] == SAMPLE_SESSION_ID

    def test_provided_session_id_skips_session_creation(self, api_client, mock_rag_system):
        """When session_id is supplied, create_session() is NOT called."""
        api_client.post(
            "/api/query",
            json={"query": "Follow up", "session_id": "existing-session"},
        )
        mock_rag_system.session_manager.create_session.assert_not_called()

    def test_provided_session_id_forwarded_to_rag(self, api_client, mock_rag_system):
        """The supplied session_id is passed directly to rag_system.query()."""
        api_client.post(
            "/api/query",
            json={"query": "test question", "session_id": "my-session"},
        )
        mock_rag_system.query.assert_called_once_with("test question", "my-session")

    def test_query_text_forwarded_to_rag(self, api_client, mock_rag_system):
        """The query string is passed to rag_system.query() as the first argument."""
        api_client.post("/api/query", json={"query": "explain neural networks"})
        call_args = mock_rag_system.query.call_args[0]
        assert call_args[0] == "explain neural networks"

    def test_missing_body_returns_422(self, api_client):
        """A request with no body returns 422 Unprocessable Entity."""
        response = api_client.post("/api/query")
        assert response.status_code == 422

    def test_missing_query_field_returns_422(self, api_client):
        """A body missing the required 'query' field returns 422."""
        response = api_client.post("/api/query", json={"session_id": "abc"})
        assert response.status_code == 422

    def test_rag_exception_returns_500(self, api_client, mock_rag_system):
        """When rag_system.query() raises, the endpoint returns HTTP 500."""
        mock_rag_system.query.side_effect = RuntimeError("Unexpected failure")
        response = api_client.post("/api/query", json={"query": "test"})
        assert response.status_code == 500

    def test_rag_exception_detail_in_500_response(self, api_client, mock_rag_system):
        """The exception message appears in the 500 response's 'detail' field."""
        mock_rag_system.query.side_effect = RuntimeError("Internal error occurred")
        response = api_client.post("/api/query", json={"query": "test"})
        assert "Internal error occurred" in response.json()["detail"]

    def test_session_creation_error_returns_500(self, api_client, mock_rag_system):
        """If create_session() itself raises, the endpoint returns 500."""
        mock_rag_system.session_manager.create_session.side_effect = RuntimeError(
            "Session store unavailable"
        )
        response = api_client.post("/api/query", json={"query": "test"})
        assert response.status_code == 500


# ---------------------------------------------------------------------------
# GET /api/courses
# ---------------------------------------------------------------------------

class TestCoursesEndpoint:

    def test_courses_returns_200(self, api_client):
        """GET /api/courses returns HTTP 200 on success."""
        response = api_client.get("/api/courses")
        assert response.status_code == 200

    def test_courses_returns_total_count(self, api_client):
        """Response body includes the correct total_courses count."""
        response = api_client.get("/api/courses")
        assert response.json()["total_courses"] == SAMPLE_COURSES["total_courses"]

    def test_courses_returns_title_list(self, api_client):
        """Response body includes the correct course_titles list."""
        response = api_client.get("/api/courses")
        assert response.json()["course_titles"] == SAMPLE_COURSES["course_titles"]

    def test_courses_delegates_to_get_course_analytics(self, api_client, mock_rag_system):
        """Endpoint calls rag_system.get_course_analytics() exactly once."""
        api_client.get("/api/courses")
        mock_rag_system.get_course_analytics.assert_called_once()

    def test_courses_empty_catalog(self, api_client, mock_rag_system):
        """When analytics returns no courses, response reflects zero count and empty list."""
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": [],
        }
        response = api_client.get("/api/courses")
        data = response.json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []

    def test_courses_exception_returns_500(self, api_client, mock_rag_system):
        """When get_course_analytics() raises, the endpoint returns HTTP 500."""
        mock_rag_system.get_course_analytics.side_effect = RuntimeError("DB error")
        response = api_client.get("/api/courses")
        assert response.status_code == 500

    def test_courses_500_includes_error_detail(self, api_client, mock_rag_system):
        """The exception message appears in the 500 response's 'detail' field."""
        mock_rag_system.get_course_analytics.side_effect = RuntimeError("Connection lost")
        response = api_client.get("/api/courses")
        assert "Connection lost" in response.json()["detail"]
