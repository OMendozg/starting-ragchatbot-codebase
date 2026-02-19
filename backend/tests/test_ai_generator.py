"""
Tests for AIGenerator in ai_generator.py

These tests verify:
- Tools are forwarded correctly to the Anthropic API
- A tool_use response triggers tool execution via the tool_manager
- Tool results (including error strings) are sent back to Claude in the follow-up call
- API errors (BadRequestError, AuthenticationError, etc.) are handled or propagate
  in a way that surfaces meaningful information to callers
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import MagicMock, patch, call
import anthropic

from ai_generator import AIGenerator


# ---------------------------------------------------------------------------
# Helpers / factories
# ---------------------------------------------------------------------------

def _make_generator():
    """Return an AIGenerator with a mocked Anthropic client."""
    with patch("anthropic.Anthropic") as MockClient:
        gen = AIGenerator(api_key="test-key", model="claude-test-model")
        # Replace the client instance that was created during __init__
        gen.client = MockClient.return_value
        return gen


def _make_tool_use_block(name, input_dict, block_id="tool_abc"):
    block = MagicMock()
    block.type = "tool_use"
    block.name = name
    block.id = block_id
    block.input = input_dict
    return block


def _make_text_block(text):
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _make_response(stop_reason, content_blocks):
    resp = MagicMock()
    resp.stop_reason = stop_reason
    resp.content = content_blocks
    return resp


TOOL_DEF = {
    "name": "search_course_content",
    "description": "Search course materials",
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
}


# ---------------------------------------------------------------------------
# Direct (no-tool) response path
# ---------------------------------------------------------------------------

class TestDirectResponse:

    def test_direct_text_response_returned(self):
        """When stop_reason is end_turn, generate_response returns text directly."""
        gen = _make_generator()
        gen.client.messages.create.return_value = _make_response(
            "end_turn", [_make_text_block("Hello world.")]
        )

        result = gen.generate_response(query="Hi")

        assert result == "Hello world."

    def test_no_tools_in_api_call_when_not_supplied(self):
        """generate_response omits 'tools' key when tools=None."""
        gen = _make_generator()
        gen.client.messages.create.return_value = _make_response(
            "end_turn", [_make_text_block("OK")]
        )

        gen.generate_response(query="Hi")

        kwargs = gen.client.messages.create.call_args[1]
        assert "tools" not in kwargs
        assert "tool_choice" not in kwargs

    def test_tools_included_in_api_call(self):
        """generate_response includes tools and tool_choice when tools are provided."""
        gen = _make_generator()
        gen.client.messages.create.return_value = _make_response(
            "end_turn", [_make_text_block("OK")]
        )

        gen.generate_response(query="Hi", tools=[TOOL_DEF])

        kwargs = gen.client.messages.create.call_args[1]
        assert kwargs["tools"] == [TOOL_DEF]
        assert kwargs["tool_choice"] == {"type": "auto"}

    def test_conversation_history_added_to_system(self):
        """generate_response appends conversation_history to the system prompt."""
        gen = _make_generator()
        gen.client.messages.create.return_value = _make_response(
            "end_turn", [_make_text_block("OK")]
        )

        gen.generate_response(
            query="Follow-up question",
            conversation_history="User: Hello\nAssistant: Hi",
        )

        kwargs = gen.client.messages.create.call_args[1]
        assert "Previous conversation:" in kwargs["system"]
        assert "User: Hello" in kwargs["system"]


# ---------------------------------------------------------------------------
# Tool-use path
# ---------------------------------------------------------------------------

class TestToolUsePath:

    def test_tool_use_triggers_tool_execution(self):
        """When stop_reason is tool_use, execute_tool is called on the tool_manager."""
        gen = _make_generator()
        tool_block = _make_tool_use_block("search_course_content", {"query": "API basics"})
        gen.client.messages.create.side_effect = [
            _make_response("tool_use", [tool_block]),
            _make_response("end_turn", [_make_text_block("Final answer.")]),
        ]

        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "Some retrieved content."

        result = gen.generate_response(
            query="What is an API?",
            tools=[TOOL_DEF],
            tool_manager=tool_manager,
        )

        tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="API basics"
        )
        assert result == "Final answer."

    def test_tool_result_sent_back_to_claude(self):
        """The tool result string is included in the follow-up messages to Claude."""
        gen = _make_generator()
        tool_block = _make_tool_use_block("search_course_content", {"query": "test"})
        gen.client.messages.create.side_effect = [
            _make_response("tool_use", [tool_block]),
            _make_response("end_turn", [_make_text_block("Done.")]),
        ]

        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "Course content here."

        gen.generate_response(query="Q", tools=[TOOL_DEF], tool_manager=tool_manager)

        # The second API call should include the tool result in messages
        second_call_kwargs = gen.client.messages.create.call_args_list[1][1]
        messages = second_call_kwargs["messages"]
        tool_result_message = messages[-1]
        assert tool_result_message["role"] == "user"
        content = tool_result_message["content"]
        assert any(
            item.get("type") == "tool_result" and "Course content here." in item.get("content", "")
            for item in content
        )

    def test_tool_id_preserved_in_result(self):
        """tool_use_id in the tool_result matches the original tool block id."""
        gen = _make_generator()
        tool_block = _make_tool_use_block(
            "search_course_content", {"query": "test"}, block_id="tool_xyz"
        )
        gen.client.messages.create.side_effect = [
            _make_response("tool_use", [tool_block]),
            _make_response("end_turn", [_make_text_block("Done.")]),
        ]

        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "Result."

        gen.generate_response(query="Q", tools=[TOOL_DEF], tool_manager=tool_manager)

        second_call_kwargs = gen.client.messages.create.call_args_list[1][1]
        messages = second_call_kwargs["messages"]
        tool_result_message = messages[-1]
        content = tool_result_message["content"]
        assert any(item.get("tool_use_id") == "tool_xyz" for item in content)

    def test_follow_up_call_omits_tools(self):
        """The second API call (after tool execution) does NOT include 'tools'."""
        gen = _make_generator()
        tool_block = _make_tool_use_block("search_course_content", {"query": "test"})
        gen.client.messages.create.side_effect = [
            _make_response("tool_use", [tool_block]),
            _make_response("end_turn", [_make_text_block("Done.")]),
        ]

        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "Result."

        gen.generate_response(query="Q", tools=[TOOL_DEF], tool_manager=tool_manager)

        second_call_kwargs = gen.client.messages.create.call_args_list[1][1]
        assert "tools" not in second_call_kwargs


# ---------------------------------------------------------------------------
# Error string propagation  (THIS REVEALS THE BUG)
# ---------------------------------------------------------------------------

class TestErrorStringPropagation:

    def test_search_error_string_from_tool_reaches_claude(self):
        """
        KNOWN ISSUE: When CourseSearchTool.execute() returns a raw error string
        (e.g. 'Search error: Number of requested results 5 > index size 2'),
        that string is sent verbatim to Claude as the tool result.
        Claude then responds with a user-facing 'query failed' type message.

        This test documents the current (broken) behavior.
        """
        gen = _make_generator()
        tool_block = _make_tool_use_block("search_course_content", {"query": "test"})
        gen.client.messages.create.side_effect = [
            _make_response("tool_use", [tool_block]),
            _make_response("end_turn", [_make_text_block("The query failed due to an internal error.")]),
        ]

        tool_manager = MagicMock()
        # Tool returns a raw ChromaDB error string
        tool_manager.execute_tool.return_value = (
            "Search error: Number of requested results 5 is greater than "
            "number of elements in index 2"
        )

        result = gen.generate_response(query="Q", tools=[TOOL_DEF], tool_manager=tool_manager)

        # Verify the error string was passed to Claude (second API call)
        second_call_messages = gen.client.messages.create.call_args_list[1][1]["messages"]
        tool_result_content = second_call_messages[-1]["content"]
        raw_error_sent = any(
            "Search error" in item.get("content", "") for item in tool_result_content
        )
        assert raw_error_sent, (
            "The raw Search error string was not forwarded to Claude — "
            "check the tool result assembly in _handle_tool_execution"
        )

        # And the final user-facing response reflects the failure
        assert "failed" in result.lower() or "error" in result.lower()


# ---------------------------------------------------------------------------
# API-level error handling  (THIS IS THE ROOT CAUSE)
# ---------------------------------------------------------------------------

class TestAPIErrorHandling:

    def test_bad_request_error_returns_friendly_message(self):
        """
        FIXED: When the Anthropic API returns a 400 BadRequestError
        (e.g. 'credit balance too low'), generate_response() catches it and
        returns a user-friendly string instead of raising.
        """
        gen = _make_generator()
        mock_resp = MagicMock(status_code=400)
        gen.client.messages.create.side_effect = anthropic.BadRequestError(
            message="Your credit balance is too low",
            response=mock_resp,
            body={"type": "error", "error": {"type": "invalid_request_error",
                                              "message": "credit balance too low"}},
        )

        result = gen.generate_response(query="What is Python?", tools=[TOOL_DEF])

        assert isinstance(result, str)
        assert len(result) > 0
        # Should mention credit/billing, not a raw exception traceback
        assert "credit" in result.lower() or "account" in result.lower()

    def test_authentication_error_returns_friendly_message(self):
        """
        An invalid API key raises AuthenticationError; generate_response()
        catches it and returns a user-friendly string.
        """
        gen = _make_generator()
        mock_resp = MagicMock(status_code=401)
        gen.client.messages.create.side_effect = anthropic.AuthenticationError(
            message="Invalid API key",
            response=mock_resp,
            body={"type": "error"},
        )

        result = gen.generate_response(query="What is Python?")

        assert isinstance(result, str)
        assert "api key" in result.lower() or "key" in result.lower()

    def test_rate_limit_error_returns_friendly_message(self):
        """
        A 429 RateLimitError is caught and returns a user-friendly string.
        """
        gen = _make_generator()
        mock_resp = MagicMock(status_code=429)
        gen.client.messages.create.side_effect = anthropic.RateLimitError(
            message="Rate limit exceeded",
            response=mock_resp,
            body={"type": "error"},
        )

        result = gen.generate_response(query="What is Python?")

        assert isinstance(result, str)
        assert "rate" in result.lower() or "wait" in result.lower()
