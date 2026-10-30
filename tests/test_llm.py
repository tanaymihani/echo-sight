import httpx
import pytest
from openai import BadRequestError

from echosight.config import LLMConfig
from echosight.llm import LLM


def bad_request(message):
    response = httpx.Response(400, request=httpx.Request("POST", "https://api.example.com"))
    return BadRequestError(message, response=response, body=None)


class FakeCompletions:
    def __init__(self, reject):
        self.reject, self.requests = list(reject), []

    def create(self, **request):
        self.requests.append(request)
        if self.reject:
            raise bad_request(self.reject.pop(0))
        return []


def llm_with(completions, model="openai/gpt-oss-120b"):
    llm = LLM(LLMConfig(provider="groq", model=model, base_url="https://api.example.com", api_key="k"))
    llm.client = type("Client", (), {"chat": type("Chat", (), {"completions": completions})})()
    return llm


def test_reasoning_settings_per_model():
    assert llm_with(FakeCompletions([]))._reasoning_params() == {"reasoning_effort": "low",
                                                                  "include_reasoning": False}
    assert llm_with(FakeCompletions([]), "qwen/qwen3.8-27b")._reasoning_params() == {"reasoning_effort": "none"}


def test_rejected_optional_parameter_is_dropped_and_remembered():
    completions = FakeCompletions(["property 'include_reasoning' is unsupported"])
    llm = llm_with(completions)
    llm.chat([{"role": "user", "content": "hi"}])
    assert completions.requests[0]["extra_body"] == {"reasoning_effort": "low", "include_reasoning": False}
    assert completions.requests[1]["extra_body"] == {"reasoning_effort": "low"}
    llm.chat([{"role": "user", "content": "again"}])
    assert completions.requests[2]["extra_body"] == {"reasoning_effort": "low"}


def test_other_bad_requests_still_raise():
    llm = llm_with(FakeCompletions(["messages: field required"]))
    with pytest.raises(BadRequestError):
        llm.chat([])
