"""
Shared pytest fixtures for the backend test suite.

Provides:
- mock_config:      MagicMock config with sensible test defaults.
- mock_rag_system:  MagicMock RAGSystem pre-configured with typical responses.
- api_client:       Starlette TestClient backed by a minimal FastAPI app that
                    mirrors /api/query and /api/courses from app.py — without
                    the static-file mount that fails in test environments.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import MagicMock
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel
from typing import List, Optional


from helpers import SAMPLE_ANSWER, SAMPLE_SOURCES, SAMPLE_SESSION_ID, SAMPLE_COURSES


# ---------------------------------------------------------------------------
# Config fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_config():
    """MagicMock config with sensible test defaults."""
    cfg = MagicMock()
    cfg.ANTHROPIC_API_KEY = "test-key"
    cfg.ANTHROPIC_MODEL = "claude-test-model"
    cfg.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    cfg.CHROMA_PATH = "/tmp/test_chroma"
    cfg.CHUNK_SIZE = 800
    cfg.CHUNK_OVERLAP = 100
    cfg.MAX_RESULTS = 5
    cfg.MAX_HISTORY = 2
    return cfg


# ---------------------------------------------------------------------------
# RAGSystem fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_rag_system():
    """
    MagicMock RAGSystem pre-configured with typical successful responses.
    Override individual attributes in tests to simulate failures or edge cases.
    """
    rag = MagicMock()
    rag.query.return_value = (SAMPLE_ANSWER, SAMPLE_SOURCES)
    rag.get_course_analytics.return_value = SAMPLE_COURSES
    rag.session_manager.create_session.return_value = SAMPLE_SESSION_ID
    return rag


# ---------------------------------------------------------------------------
# API client fixture
# ---------------------------------------------------------------------------

def _build_test_app(rag_system: MagicMock) -> FastAPI:
    """
    Build a minimal FastAPI app that mirrors /api/query and /api/courses from
    app.py, but without the static-file mount so tests can import it safely.
    """
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[str]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    test_app = FastAPI(title="Test RAG App")

    @test_app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = rag_system.session_manager.create_session()
            answer, sources = rag_system.query(request.query, session_id)
            return QueryResponse(answer=answer, sources=sources, session_id=session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @test_app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"],
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return test_app


@pytest.fixture
def api_client(mock_rag_system):
    """
    Starlette TestClient wrapping a minimal test FastAPI app.

    The app mirrors /api/query and /api/courses from app.py, but avoids the
    static-file mount so no frontend directory is required during tests.
    Shares the same mock_rag_system instance as other fixtures in the same
    test, so you can configure the mock before making requests.
    """
    app = _build_test_app(mock_rag_system)
    with TestClient(app) as client:
        yield client
