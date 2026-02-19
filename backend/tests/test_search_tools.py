"""
Tests for CourseSearchTool.execute() in search_tools.py

These tests verify that the search tool correctly:
- Delegates queries to the vector store
- Returns formatted results
- Returns meaningful error/empty messages
- Tracks sources after each search
- Handles error results from the vector store
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import MagicMock, call
from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_store():
    return MagicMock()


@pytest.fixture
def tool(mock_store):
    return CourseSearchTool(mock_store)


def make_results(docs, metas, error=None):
    """Helper to build a SearchResults object."""
    if error:
        return SearchResults.empty(error)
    return SearchResults(
        documents=docs,
        metadata=metas,
        distances=[0.1] * len(docs),
    )


# ---------------------------------------------------------------------------
# execute() – happy path
# ---------------------------------------------------------------------------

class TestCourseSearchToolExecuteHappyPath:

    def test_returns_formatted_content(self, tool, mock_store):
        """execute() returns document text wrapped with course/lesson header."""
        mock_store.search.return_value = make_results(
            docs=["Python is a high-level language."],
            metas=[{"course_title": "Python Basics", "lesson_number": 1}],
        )
        mock_store.get_lesson_link.return_value = "https://example.com/lesson1"

        result = tool.execute(query="What is Python?")

        assert "Python Basics" in result
        assert "Lesson 1" in result
        assert "Python is a high-level language." in result

    def test_multiple_results_all_included(self, tool, mock_store):
        """execute() includes all returned documents in the output."""
        mock_store.search.return_value = make_results(
            docs=["Doc A content.", "Doc B content."],
            metas=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course B", "lesson_number": 2},
            ],
        )
        mock_store.get_lesson_link.return_value = None

        result = tool.execute(query="general topic")

        assert "Doc A content." in result
        assert "Doc B content." in result

    def test_result_without_lesson_number(self, tool, mock_store):
        """execute() works when metadata has no lesson_number."""
        mock_store.search.return_value = make_results(
            docs=["Intro content."],
            metas=[{"course_title": "Intro Course"}],
        )

        result = tool.execute(query="intro")

        assert "Intro Course" in result
        assert "Lesson" not in result


# ---------------------------------------------------------------------------
# execute() – filter delegation
# ---------------------------------------------------------------------------

class TestCourseSearchToolFilterDelegation:

    def test_query_only_passed_to_store(self, tool, mock_store):
        """execute() passes query-only call correctly to vector store."""
        mock_store.search.return_value = make_results([], [])

        tool.execute(query="some query")

        mock_store.search.assert_called_once_with(
            query="some query",
            course_name=None,
            lesson_number=None,
        )

    def test_course_name_filter_passed_to_store(self, tool, mock_store):
        """execute() forwards course_name filter to vector store."""
        mock_store.search.return_value = make_results([], [])

        tool.execute(query="test", course_name="Python Basics")

        mock_store.search.assert_called_once_with(
            query="test",
            course_name="Python Basics",
            lesson_number=None,
        )

    def test_lesson_number_filter_passed_to_store(self, tool, mock_store):
        """execute() forwards lesson_number filter to vector store."""
        mock_store.search.return_value = make_results([], [])

        tool.execute(query="test", lesson_number=3)

        mock_store.search.assert_called_once_with(
            query="test",
            course_name=None,
            lesson_number=3,
        )

    def test_both_filters_passed_to_store(self, tool, mock_store):
        """execute() forwards both course_name and lesson_number to vector store."""
        mock_store.search.return_value = make_results([], [])

        tool.execute(query="test", course_name="ML Course", lesson_number=2)

        mock_store.search.assert_called_once_with(
            query="test",
            course_name="ML Course",
            lesson_number=2,
        )


# ---------------------------------------------------------------------------
# execute() – empty / not-found results
# ---------------------------------------------------------------------------

class TestCourseSearchToolEmptyResults:

    def test_no_results_returns_not_found_message(self, tool, mock_store):
        """execute() returns a clear 'not found' string on empty results."""
        mock_store.search.return_value = make_results([], [])

        result = tool.execute(query="obscure topic")

        assert "No relevant content found" in result

    def test_no_results_with_course_filter_includes_course_in_message(self, tool, mock_store):
        """execute() includes the course name in the not-found message."""
        mock_store.search.return_value = make_results([], [])

        result = tool.execute(query="topic", course_name="Advanced ML")

        assert "Advanced ML" in result

    def test_no_results_with_lesson_filter_includes_lesson_in_message(self, tool, mock_store):
        """execute() includes the lesson number in the not-found message."""
        mock_store.search.return_value = make_results([], [])

        result = tool.execute(query="topic", lesson_number=5)

        assert "5" in result


# ---------------------------------------------------------------------------
# execute() – error propagation  (THIS IS WHERE THE BUG SURFACES)
# ---------------------------------------------------------------------------

class TestCourseSearchToolErrorHandling:

    def test_search_error_is_returned_as_string(self, tool, mock_store):
        """
        KNOWN ISSUE: When the vector store returns an error (e.g. n_results >
        collection size), execute() returns the raw error string directly.
        This string is then sent to Claude as the tool result, causing Claude
        to reply with something like 'the query failed'.
        """
        error_msg = (
            "Search error: Number of requested results 5 is greater than "
            "number of elements in index 2"
        )
        mock_store.search.return_value = make_results([], [], error=error_msg)

        result = tool.execute(query="anything")

        # The raw ChromaDB error leaks to the caller / Claude
        assert "Search error" in result

    def test_course_not_found_error_returned_as_string(self, tool, mock_store):
        """
        When a course name can't be resolved, a 'No course found' error
        string is returned. This again propagates to Claude verbatim.
        """
        mock_store.search.return_value = make_results(
            [], [], error="No course found matching 'NonExistent'"
        )

        result = tool.execute(query="test", course_name="NonExistent")

        assert "No course found" in result


# ---------------------------------------------------------------------------
# execute() – source tracking
# ---------------------------------------------------------------------------

class TestCourseSearchToolSourceTracking:

    def test_sources_tracked_after_search(self, tool, mock_store):
        """execute() populates last_sources with one entry per result."""
        mock_store.search.return_value = make_results(
            docs=["Content."],
            metas=[{"course_title": "Test Course", "lesson_number": 1}],
        )
        mock_store.get_lesson_link.return_value = "https://example.com"

        tool.execute(query="test")

        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["label"] == "Test Course - Lesson 1"
        assert tool.last_sources[0]["url"] == "https://example.com"

    def test_sources_cleared_between_calls(self, tool, mock_store):
        """last_sources reflects only the most recent search."""
        mock_store.search.return_value = make_results(
            docs=["Doc 1.", "Doc 2."],
            metas=[
                {"course_title": "A", "lesson_number": 1},
                {"course_title": "B", "lesson_number": 2},
            ],
        )
        mock_store.get_lesson_link.return_value = None
        tool.execute(query="first")

        mock_store.search.return_value = make_results(
            docs=["Doc X."],
            metas=[{"course_title": "X", "lesson_number": 9}],
        )
        tool.execute(query="second")

        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["label"] == "X - Lesson 9"

    def test_no_sources_on_empty_results(self, tool, mock_store):
        """execute() leaves last_sources empty when there are no results."""
        mock_store.search.return_value = make_results([], [])

        tool.execute(query="nothing")

        assert tool.last_sources == []

    def test_source_url_is_none_when_link_unavailable(self, tool, mock_store):
        """execute() sets url=None when the store has no lesson link."""
        mock_store.search.return_value = make_results(
            docs=["Content."],
            metas=[{"course_title": "Course", "lesson_number": 1}],
        )
        mock_store.get_lesson_link.return_value = None

        tool.execute(query="test")

        assert tool.last_sources[0]["url"] is None


# ---------------------------------------------------------------------------
# ToolManager integration
# ---------------------------------------------------------------------------

class TestToolManager:

    def test_register_and_execute_tool(self, mock_store):
        """ToolManager routes execute_tool() to the correct registered tool."""
        tool = CourseSearchTool(mock_store)
        mock_store.search.return_value = make_results(
            docs=["Answer content."],
            metas=[{"course_title": "C", "lesson_number": 1}],
        )
        mock_store.get_lesson_link.return_value = None

        manager = ToolManager()
        manager.register_tool(tool)

        result = manager.execute_tool("search_course_content", query="test")

        assert "Answer content." in result

    def test_get_last_sources_from_manager(self, mock_store):
        """ToolManager.get_last_sources() returns sources set by the search tool."""
        tool = CourseSearchTool(mock_store)
        mock_store.search.return_value = make_results(
            docs=["Text."],
            metas=[{"course_title": "D", "lesson_number": 2}],
        )
        mock_store.get_lesson_link.return_value = "https://link.com"

        manager = ToolManager()
        manager.register_tool(tool)
        manager.execute_tool("search_course_content", query="query")

        sources = manager.get_last_sources()
        assert len(sources) == 1
        assert sources[0]["label"] == "D - Lesson 2"

    def test_reset_sources_clears_state(self, mock_store):
        """ToolManager.reset_sources() clears last_sources on all tools."""
        tool = CourseSearchTool(mock_store)
        mock_store.search.return_value = make_results(
            docs=["Text."],
            metas=[{"course_title": "E", "lesson_number": 3}],
        )
        mock_store.get_lesson_link.return_value = None

        manager = ToolManager()
        manager.register_tool(tool)
        manager.execute_tool("search_course_content", query="query")

        manager.reset_sources()

        assert manager.get_last_sources() == []

    def test_unknown_tool_returns_error_string(self):
        """ToolManager returns an error string for unregistered tool names."""
        manager = ToolManager()
        result = manager.execute_tool("nonexistent_tool", query="test")
        assert "not found" in result.lower()
