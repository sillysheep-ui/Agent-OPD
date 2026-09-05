from omniopd.adapters import extract_task_description, extract_task_type, unwrap_batched_info


def test_task_extraction_does_not_absorb_the_rest_of_the_initial_observation():
    observation = (
        "You arrive in a kitchen. Your task is to: cool a tomato and put it away. "
        "The fridge is open and a tomato is visible."
    )
    assert extract_task_description(observation) == "cool a tomato and put it away"


def test_batched_environment_info_is_unwrapped_consistently():
    info = {"won": [False], "admissible_commands": [["look", "go"]], "label": "x"}
    assert unwrap_batched_info(info) == {
        "won": False,
        "admissible_commands": ["look", "go"],
        "label": "x",
    }


def test_task_type_comes_from_task_family_directory_not_trial_or_split():
    game = (
        "/data/json_2.1.1/train/"
        "pick_and_place_simple-Apple-None-Desk-10/"
        "trial_T20190908_125747_237623/game.tw-pddl"
    )
    assert extract_task_type(game) == "pick_and_place_simple"


def test_game_listing_passes_the_exact_alfworld_split_name():
    import sys
    import types

    from omniopd.adapters import list_alfworld_games

    observed = []

    class FakeAlfredTWEnv:
        def __init__(self, config, train_eval):
            observed.append(train_eval)
            self.game_files = [f"/{train_eval}/game.z8"]

    names = [
        "alfworld",
        "alfworld.agents",
        "alfworld.agents.environment",
        "alfworld.agents.environment.alfred_tw_env",
    ]
    previous = {name: sys.modules.get(name) for name in names}
    try:
        for name in names:
            sys.modules[name] = types.ModuleType(name)
        sys.modules[names[-1]].AlfredTWEnv = FakeAlfredTWEnv
        games = list_alfworld_games({}, "eval_out_of_distribution")
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    assert observed == ["eval_out_of_distribution"]
    assert games == ["/eval_out_of_distribution/game.z8"]


def test_openai_adapter_sends_explicit_thinking_contract():
    import sys
    import types

    from omniopd.adapters import OpenAIChatPolicy

    calls = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            message = types.SimpleNamespace(content="Action: look")
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = types.SimpleNamespace(completions=Completions())

    previous = sys.modules.get("openai")
    module = types.ModuleType("openai")
    module.OpenAI = FakeOpenAI
    sys.modules["openai"] = module
    try:
        nonthinking = OpenAIChatPolicy(
            model="teacher",
            base_url="https://example.test",
            api_key="secret",
            thinking_mode="disabled",
        )
        nonthinking.generate([], temperature=0.7, max_tokens=8, request_id="a")
        thinking = OpenAIChatPolicy(
            model="teacher",
            base_url="https://example.test",
            api_key="secret",
            thinking_mode="enabled",
            reasoning_effort="high",
        )
        thinking.generate([], temperature=None, max_tokens=8, request_id="b")
    finally:
        if previous is None:
            sys.modules.pop("openai", None)
        else:
            sys.modules["openai"] = previous
    assert calls[0]["temperature"] == 0.7
    assert calls[0]["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "temperature" not in calls[1]
    assert calls[1]["reasoning_effort"] == "high"
    assert calls[1]["extra_body"] == {"thinking": {"type": "enabled"}}
    assert nonthinking.request_ledger[0]["request_id"] == "a"
    assert nonthinking.request_ledger[0]["sdk_max_retries"] == 0
    assert nonthinking.request_ledger[0]["status"] == "ok"


def test_vllm_qwen_uses_chat_template_thinking_switch():
    import sys
    import types

    from omniopd.adapters import OpenAIChatPolicy

    calls = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            message = types.SimpleNamespace(content="Action: look")
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    class FakeOpenAI:
        def __init__(self, **kwargs):
            assert kwargs["max_retries"] == 0
            self.chat = types.SimpleNamespace(completions=Completions())

    previous = sys.modules.get("openai")
    module = types.ModuleType("openai")
    module.OpenAI = FakeOpenAI
    sys.modules["openai"] = module
    try:
        policy = OpenAIChatPolicy(
            model="student",
            base_url="http://localhost/v1",
            api_key="EMPTY",
            thinking_mode="disabled",
            thinking_control="chat_template",
        )
        policy.generate([], temperature=0.0, max_tokens=8, request_id="student:0")
    finally:
        if previous is None:
            sys.modules.pop("openai", None)
        else:
            sys.modules["openai"] = previous
    assert calls[0]["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


def test_request_error_is_immediately_persisted_without_sdk_retry():
    import json
    import sys
    import tempfile
    import types
    from pathlib import Path

    from omniopd.adapters import OpenAIChatPolicy

    class Completions:
        def create(self, **kwargs):
            raise TimeoutError("simulated")

    class FakeOpenAI:
        def __init__(self, **kwargs):
            assert kwargs["max_retries"] == 0
            self.chat = types.SimpleNamespace(completions=Completions())

    previous = sys.modules.get("openai")
    module = types.ModuleType("openai")
    module.OpenAI = FakeOpenAI
    sys.modules["openai"] = module
    try:
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / "requests.partial.jsonl"
            policy = OpenAIChatPolicy(
                model="teacher",
                base_url="https://example.test",
                api_key="secret",
                thinking_mode="disabled",
                request_ledger_path=ledger,
            )
            try:
                policy.generate([], temperature=1.0, max_tokens=8, request_id="failed")
            except TimeoutError:
                pass
            else:
                raise AssertionError("the simulated request must fail")
            row = json.loads(ledger.read_text(encoding="utf-8"))
            assert row["status"] == "error"
            assert row["error_type"] == "TimeoutError"
    finally:
        if previous is None:
            sys.modules.pop("openai", None)
        else:
            sys.modules["openai"] = previous
