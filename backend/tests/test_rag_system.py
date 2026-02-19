"""
Tests for how RAGSystem handles content-query related questions.

These tests verify:
- The full query() pipeline: prompt building → AI generation → source retrieval
- That API errors from the AI generator result in meaningful failures (not 'query failed')
- That sources from the tool are correctly retrieved and returned
- That conversation history is correctly passed to the AI generator
- That sources are reset after each query
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import MagicMock, patch, call
import anthropic

from rag_system import RAGSystem
from vector_store import SearchResults


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config():
    cfg = MagicMock()
    cfg.ANTHROPIC_API_KEY = "test-key"
    cfg.ANTHROPIC_MODEL = "claude-test"
    cfg.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    cfg.CHROMA_PATH = "/tmp/test_chroma"
    cfg.CHUNK_SIZE = 800
    cfg.CHUNK_OVERLAP = 100
    cfg.MAX_RESULTS = 5
    cfg.MAX_HISTORY = 2
    return cfg


@pytest.fixture
def rag(tmp_path):
    """
    RAGSystem with all heavy dependencies mocked so no real ChromaDB,
    no real embedding model, and no real Anthropic calls are needed.
    """
    with patch("rag_system.VectorStore") as MockVS, \
         patch("rag_system.AIGenerator") as MockAI, \
         patch("rag_system.DocumentProcessor"), \
         patch("rag_system.SessionManager") as MockSM:

        cfg = _make_config()
        cfg.CHROMA_PATH = str(tmp_path / "chroma")

        system = RAGSystem(cfg)

        # Expose mocks for test assertions
        system._mock_vector_store = MockVS.return_value
        system._mock_ai_generator = MockAI.return_value
        system._mock_session_manager = MockSM.return_value

        yield system


# ---------------------------------------------------------------------------
# query() – prompt construction
# ---------------------------------------------------------------------------

class TestRAGSystemPromptConstruction:

    def test_query_wraps_user_question_in_prompt(self, rag):
        """RAGSystem.query() wraps the user's question in the course-materials prompt."""
        rag._mock_ai_generator.generate_response.return_value = "Some answer."
        rag._mock_session_manager.get_conversation_history.return_value = None

        rag.query("What is machine learning?", session_id=None)

        call_kwargs = rag._mock_ai_generator.generate_response.call_args[1]
        assert "machine learning" in call_kwargs["query"]
        assert "course materials" in call_kwargs["query"].lower()

    def test_query_passes_tool_definitions(self, rag):
        """RAGSystem.query() always supplies tool definitions to the AI generator."""
        rag._mock_ai_generator.generate_response.return_value = "Answer."
        rag._mock_session_manager.get_conversation_history.return_value = None

        rag.query("test question", session_id=None)

        call_kwargs = rag._mock_ai_generator.generate_response.call_args[1]
        assert call_kwargs["tools"] is not None
        assert len(call_kwargs["tools"]) > 0

    def test_query_passes_tool_manager(self, rag):
        """RAGSystem.query() passes the tool_manager to the AI generator."""
        rag._mock_ai_generator.generate_response.return_value = "Answer."
        rag._mock_session_manager.get_conversation_history.return_value = None

        rag.query("test question", session_id=None)

        call_kwargs = rag._mock_ai_generator.generate_response.call_args[1]
        assert call_kwargs["tool_manager"] is rag.tool_manager


# ---------------------------------------------------------------------------
# query() – conversation history
# ---------------------------------------------------------------------------

class TestRAGSystemConversationHistory:

    def test_history_fetched_for_known_session(self, rag):
        """RAGSystem.query() retrieves conversation history for a known session_id."""
        rag._mock_session_manager.get_conversation_history.return_value = (
            "User: hello\nAssistant: hi"
        )
        rag._mock_ai_generator.generate_response.return_value = "Answer."

        rag.query("follow-up question", session_id="session_1")

        rag._mock_session_manager.get_conversation_history.assert_called_once_with(
            "session_1"
        )

    def test_history_passed_to_ai_generator(self, rag):
        """The retrieved history is forwarded to AIGenerator.generate_response()."""
        history = "User: prev\nAssistant: prev answer"
        rag._mock_session_manager.get_conversation_history.return_value = history
        rag._mock_ai_generator.generate_response.return_value = "Answer."

        rag.query("next question", session_id="session_1")

        call_kwargs = rag._mock_ai_generator.generate_response.call_args[1]
        assert call_kwargs["conversation_history"] == history

    def test_no_history_call_when_no_session(self, rag):
        """RAGSystem.query() skips history lookup when session_id is None."""
        rag._mock_ai_generator.generate_response.return_value = "Answer."

        rag.query("any question", session_id=None)

        rag._mock_session_manager.get_conversation_history.assert_not_called()

    def test_exchange_saved_to_session_after_query(self, rag):
        """RAGSystem.query() saves the Q&A exchange to the session manager."""
        rag._mock_session_manager.get_conversation_history.return_value = None
        rag._mock_ai_generator.generate_response.return_value = "The answer."

        rag.query("the question", session_id="session_1")

        rag._mock_session_manager.add_exchange.assert_called_once_with(
            "session_1", "the question", "The answer."
        )


# ---------------------------------------------------------------------------
# query() – source retrieval and reset
# ---------------------------------------------------------------------------

class TestRAGSystemSourceHandling:

    def test_sources_returned_from_tool_manager(self, rag):
        """RAGSystem.query() returns whatever sources the tool_manager has after the call."""
        rag._mock_ai_generator.generate_response.return_value = "Answer."
        rag._mock_session_manager.get_conversation_history.return_value = None

        # Inject sources directly into the real tool (CourseSearchTool)
        rag.search_tool.last_sources = [
            {"label": "Course A - Lesson 1", "url": "https://example.com"}
        ]

        _, sources = rag.query("question", session_id=None)

        assert len(sources) == 1
        assert sources[0]["label"] == "Course A - Lesson 1"

    def test_sources_reset_after_query(self, rag):
        """RAGSystem.query() resets sources on the tool_manager after each call."""
        rag._mock_ai_generator.generate_response.return_value = "Answer."
        rag._mock_session_manager.get_conversation_history.return_value = None
        rag.search_tool.last_sources = [{"label": "X", "url": None}]

        rag.query("question", session_id=None)

        assert rag.search_tool.last_sources == []

    def test_empty_sources_when_no_search_performed(self, rag):
        """When the AI answers without using the tool, sources list is empty."""
        rag._mock_ai_generator.generate_response.return_value = "General knowledge answer."
        rag._mock_session_manager.get_conversation_history.return_value = None
        # Ensure search_tool.last_sources starts empty
        rag.search_tool.last_sources = []

        _, sources = rag.query("What year was Python created?", session_id=None)

        assert sources == []


# ---------------------------------------------------------------------------
# query() – error handling  (ROOT CAUSE OF 'query failed')
# ---------------------------------------------------------------------------

class TestRAGSystemErrorHandling:

    def test_api_bad_request_returns_friendly_string(self, rag):
        """
        FIXED: When AIGenerator returns a friendly error string (after catching
        BadRequestError internally), RAGSystem.query() returns that string as
        the answer instead of raising — so the frontend shows a real message.
        """
        rag._mock_ai_generator.generate_response.return_value = (
            "The AI service rejected the request (credit balance may be too low). "
            "Please check your Anthropic account."
        )
        rag._mock_session_manager.get_conversation_history.return_value = None

        answer, sources = rag.query("What is Python?", session_id=None)

        assert "credit" in answer.lower() or "account" in answer.lower()
        assert sources == []

    def test_general_exception_propagates_unhandled(self, rag):
        """
        Unexpected non-API exceptions (e.g. RuntimeError from a bug) still
        propagate unhandled through RAGSystem.query().
        """
        rag._mock_ai_generator.generate_response.side_effect = RuntimeError(
            "Unexpected failure"
        )
        rag._mock_session_manager.get_conversation_history.return_value = None

        with pytest.raises(RuntimeError):
            rag.query("What is Python?", session_id=None)

    def test_sources_not_saved_to_session_on_error(self, rag):
        """
        When generate_response() raises, the session exchange is NOT saved
        (add_exchange is never called), which is correct — no partial state.
        """
        rag._mock_ai_generator.generate_response.side_effect = RuntimeError("oops")
        rag._mock_session_manager.get_conversation_history.return_value = None

        try:
            rag.query("question", session_id="session_1")
        except RuntimeError:
            pass

        rag._mock_session_manager.add_exchange.assert_not_called()


# ---------------------------------------------------------------------------
# query() – successful return value shape
# ---------------------------------------------------------------------------

class TestRAGSystemReturnValue:

    def test_returns_tuple_of_answer_and_sources(self, rag):
        """RAGSystem.query() returns (answer_str, sources_list)."""
        rag._mock_ai_generator.generate_response.return_value = "The answer."
        rag._mock_session_manager.get_conversation_history.return_value = None

        result = rag.query("question", session_id=None)

        assert isinstance(result, tuple)
        assert len(result) == 2
        answer, sources = result
        assert isinstance(answer, str)
        assert isinstance(sources, list)

    def test_answer_matches_ai_generator_output(self, rag):
        """The first element of the return tuple is exactly what AIGenerator returned."""
        rag._mock_ai_generator.generate_response.return_value = "Exact AI answer."
        rag._mock_session_manager.get_conversation_history.return_value = None

        answer, _ = rag.query("question", session_id=None)

        assert answer == "Exact AI answer."
