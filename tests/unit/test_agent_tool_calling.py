"""Tests _generate_answer's manual function-calling round trip with a
fake Gemini client -- no real API calls, no quota spent. Mirrors the
shape live testing found matters: the follow-up call must not offer
tools again (found as a real bug -- the model happily calls `calculate`
a second time otherwise, leaving `.text` as None)."""

from findocqa.agent import nodes


class _FakeFunctionCall:
    def __init__(self, name, args):
        self.name = name
        self.args = args


class _FakePart:
    def __init__(self, function_call=None):
        self.function_call = function_call


class _FakeContent:
    def __init__(self, parts):
        self.parts = parts


class _FakeCandidate:
    def __init__(self, parts):
        self.content = _FakeContent(parts)


class _FakeResponse:
    def __init__(self, parts, text=None):
        self.candidates = [_FakeCandidate(parts)]
        self.text = text


class _FakeModels:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def generate_content(self, *, model, contents, config):
        self.calls.append({"contents": contents, "config": config})
        return self._responses.pop(0)


class _FakeClient:
    def __init__(self, responses):
        self.models = _FakeModels(responses)


def _patch(monkeypatch, responses):
    fake_client = _FakeClient(responses)
    monkeypatch.setattr(nodes, "_client", lambda: fake_client)
    monkeypatch.setattr(nodes, "throttle", lambda: None)
    return fake_client


def test_generate_answer_returns_text_with_no_tool_calls_when_model_answers_directly(monkeypatch):
    response = _FakeResponse(parts=[_FakePart(function_call=None)], text="3M's FY2018 capex was $1,577M.")
    _patch(monkeypatch, [response])

    answer, tool_calls = nodes._generate_answer("question", chunks=[])

    assert answer == "3M's FY2018 capex was $1,577M."
    assert tool_calls == []


def test_generate_answer_executes_calculate_and_feeds_result_back(monkeypatch):
    tool_response = _FakeResponse(
        parts=[_FakePart(function_call=_FakeFunctionCall("calculate", {"expression": "(1577+1373+1420)/3"}))]
    )
    final_response = _FakeResponse(parts=[_FakePart()], text="The 3-year average is $1,456.67M.")
    fake_client = _patch(monkeypatch, [tool_response, final_response])

    answer, tool_calls = nodes._generate_answer("question", chunks=[])

    assert answer == "The 3-year average is $1,456.67M."
    assert tool_calls == [
        {"name": "calculate", "args": {"expression": "(1577+1373+1420)/3"}, "result": 1456.6666666666667}
    ]
    # The bug found via live testing: the follow-up call must not offer
    # the tool again, or the model can call it a second time and leave
    # `.text` as None.
    follow_up_config = fake_client.models.calls[1]["config"]
    assert follow_up_config.tools is None


def test_generate_answer_records_calculator_error_without_crashing(monkeypatch):
    tool_response = _FakeResponse(
        parts=[_FakePart(function_call=_FakeFunctionCall("calculate", {"expression": "1 / 0"}))]
    )
    final_response = _FakeResponse(parts=[_FakePart()], text="I could not compute that derived figure.")
    _patch(monkeypatch, [tool_response, final_response])

    answer, tool_calls = nodes._generate_answer("question", chunks=[])

    assert answer == "I could not compute that derived figure."
    assert tool_calls == [{"name": "calculate", "args": {"expression": "1 / 0"}, "error": "Division by zero"}]
