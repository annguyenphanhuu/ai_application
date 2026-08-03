from types import SimpleNamespace

import pytest

from src.agent import (
    AgentConfig,
    PolicySource,
    ProductSearchTool,
    SmartShopAgent,
    build_langgraph_app,
    build_parser,
    format_product_hits,
)


class FakeSearchService:
    def __init__(self):
        self.queries = []

    def search_products(
        self,
        query,
        filters=None,
        top_k=5,
        category_filter=None,
    ):
        self.queries.append(
            {
                "query": query,
                "filters": filters,
                "top_k": top_k,
                "category_filter": category_filter,
            }
        )
        return [
            {
                "product_id": "P01",
                "title": "Wireless Headphones",
                "price": 149.99,
                "score": 0.91,
            }
        ]


class FakeToolCallingLLM:
    """Fake LLM that requests a tool, then answers from the tool result.

    ``answer`` is what it returns once it has seen tool output -- mirroring a
    real model, which stops emitting tool calls when it has enough to reply.
    Set ``always_call_tool=True`` to simulate a model that never stops, so the
    iteration cap can be tested.
    """

    def __init__(
        self, tool_name, args, answer="Here is what I found.", always_call_tool=False
    ):
        self.tool_name = tool_name
        self.args = args
        self.answer = answer
        self.always_call_tool = always_call_tool
        self.bound_tools = None
        self.invoke_count = 0
        self.tool_call_count = 0

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def invoke(self, messages):
        self.messages = messages
        self.invoke_count += 1
        # A synthesis prompt carries tool results; answer instead of re-calling.
        saw_tool_result = any(
            "Result of " in str(m.get("content", "")) for m in messages
        )
        if saw_tool_result and not self.always_call_tool:
            return SimpleNamespace(
                content=self.answer,
                tool_calls=[],
                usage_metadata={"input_tokens": 30, "output_tokens": 12},
                response_metadata={},
            )
        self.tool_call_count += 1
        return SimpleNamespace(
            content="",
            tool_calls=[{"name": self.tool_name, "args": self.args}],
            usage_metadata={"input_tokens": 42, "output_tokens": 7},
            response_metadata={"token_usage": {"prompt_tokens": 10}},
        )


def build_agent(require_human_approval_for_tools=False):
    service = FakeSearchService()
    agent = SmartShopAgent(
        config=AgentConfig(
            top_k=2,
            require_human_approval_for_tools=require_human_approval_for_tools,
        ),
        search_tool=ProductSearchTool(service),
    )
    return agent, service


def test_agent_answers_policy_without_calling_search_tool():
    agent, service = build_agent()

    response = agent.handle_message("What is your shipping policy?")

    assert response.action == "answer_policy"
    assert "3-5 business days" in response.content
    assert response.requires_human_review is False
    assert service.queries == []


def test_agent_searches_products_with_vector_tool():
    agent, service = build_agent()

    response = agent.handle_message("show me wireless headphones")

    assert response.action == "search_products"
    assert "Wireless Headphones" in response.content
    assert service.queries == [
        {
            "query": "show me wireless headphones",
            "filters": None,
            "top_k": 2,
            "category_filter": None,
        }
    ]
    assert response.tool_outputs[0]["tool"] == "search_products"


def test_agent_uses_llm_tool_call_for_non_keyword_product_request():
    service = FakeSearchService()
    llm = FakeToolCallingLLM(
        "search_products",
        {"query": "ergonomic desk setup", "top_k": 1},
    )
    agent = SmartShopAgent(
        config=AgentConfig(top_k=2, use_llm_routing="always"),
        search_tool=ProductSearchTool(service),
        llm=llm,
    )

    response = agent.handle_message("Can you compare comfort options for my desk?")

    assert response.action == "search_products"
    assert service.queries[0]["query"] == "ergonomic desk setup"
    assert service.queries[0]["top_k"] == 1
    assert llm.bound_tools is not None


def test_agent_emits_llm_call_trace_event_with_token_usage():
    llm = FakeToolCallingLLM(
        "search_products",
        {"query": "ergonomic desk setup", "top_k": 1},
    )
    agent = SmartShopAgent(
        config=AgentConfig(top_k=2, use_llm_routing="always"),
        search_tool=ProductSearchTool(FakeSearchService()),
        llm=llm,
    )

    response = agent.handle_message("Can you compare comfort options for my desk?")

    llm_events = [event for event in response.trace_events if event.event == "llm_call"]
    # One routing call plus one synthesis call over the tool result.
    assert [event.action for event in llm_events] == ["search_products", "synthesize"]
    assert llm_events[0].metadata["prompt_tokens"] == 42
    assert llm_events[0].metadata["completion_tokens"] == 7
    assert llm_events[0].metadata["model"]
    assert llm_events[1].metadata["prompt_tokens"] == 30


def test_llm_answers_from_tool_result_instead_of_template():
    llm = FakeToolCallingLLM(
        "search_products",
        {"query": "desk chair", "top_k": 1},
        answer="I found the Wireless Headphones at $149.99 - great value.",
    )
    agent = SmartShopAgent(
        config=AgentConfig(top_k=2, use_llm_routing="always"),
        search_tool=ProductSearchTool(FakeSearchService()),
        llm=llm,
    )

    response = agent.handle_message("something comfy for my desk?")

    # The LLM's synthesis wins over the deterministic template, which means it
    # actually saw the tool output.
    assert (
        response.content == "I found the Wireless Headphones at $149.99 - great value."
    )
    assert "Here are matching products:" not in response.content


def test_synthesis_prompt_contains_the_tool_result():
    service = FakeSearchService()
    llm = FakeToolCallingLLM("search_products", {"query": "headphones", "top_k": 1})
    agent = SmartShopAgent(
        config=AgentConfig(top_k=2, use_llm_routing="always"),
        search_tool=ProductSearchTool(service),
        llm=llm,
    )

    agent.handle_message("show me something")

    rendered = " ".join(str(m.get("content", "")) for m in llm.messages)
    assert "Result of search_products" in rendered
    assert "Wireless Headphones" in rendered


def test_agent_can_call_a_tool_again_after_seeing_results():
    service = FakeSearchService()
    llm = FakeToolCallingLLM(
        "search_products",
        {"query": "first try", "top_k": 1},
        always_call_tool=True,
    )
    agent = SmartShopAgent(
        config=AgentConfig(top_k=2, use_llm_routing="always", max_tool_iterations=3),
        search_tool=ProductSearchTool(service),
        llm=llm,
    )

    response = agent.handle_message("find me something")

    # A model that keeps requesting tools must be bounded by max_tool_iterations
    # rather than looping forever, but it does get more than one attempt.
    assert len(service.queries) == 3
    assert len(response.tool_outputs) == 3


def test_tool_failure_does_not_crash_the_turn():
    class BrokenSearchService:
        def search_products(self, query, filters=None, top_k=5, category_filter=None):
            raise RuntimeError("qdrant is down")

    agent = SmartShopAgent(
        config=AgentConfig(top_k=2, use_llm_routing="never"),
        search_tool=ProductSearchTool(BrokenSearchService()),
    )

    response = agent.handle_message("show me wireless headphones")

    assert response.action == "search_products"
    assert "could not find" in response.content.lower()
    tool_events = [e for e in response.trace_events if e.event == "tool_call"]
    assert tool_events[0].status == "error"


def test_rule_based_agent_does_not_emit_llm_call_event():
    agent, _service = build_agent()

    response = agent.handle_message("what is your shipping policy?")

    assert all(event.event != "llm_call" for event in response.trace_events)


def test_agent_retrieves_policy_from_source_file(tmp_path):
    policy_path = tmp_path / "policy.md"
    policy_path.write_text(
        "# Policies\n\n## shipping\n\nShips in 2 days from the local warehouse.\n",
        encoding="utf-8",
    )
    llm = FakeToolCallingLLM(
        "retrieve_policy",
        {"topic": "shipping", "query": "shipping timing"},
    )
    agent = SmartShopAgent(
        config=AgentConfig(
            use_llm_routing="always", policy_source_path=str(policy_path)
        ),
        policy_source=PolicySource(policy_path),
        search_tool=ProductSearchTool(FakeSearchService()),
        llm=llm,
    )

    response = agent.handle_message("When will it ship?")

    assert response.action == "answer_policy"
    # With an LLM present the reply is synthesised, so the proof that the file
    # was read is that its text reached the synthesis prompt.
    rendered = " ".join(str(m.get("content", "")) for m in llm.messages)
    assert "Ships in 2 days" in rendered
    assert response.content.startswith("Here is what I found.")
    assert str(policy_path) in response.content


def test_agent_pauses_sensitive_case_for_human_review():
    agent, service = build_agent()

    response = agent.handle_message("I want a refund for a damaged order")

    assert response.action == "request_human_review"
    assert response.requires_human_review is True
    assert "human support review" in response.content
    assert service.queries == []


def test_agent_can_require_approval_before_tool_execution():
    agent, service = build_agent(require_human_approval_for_tools=True)

    waiting = agent.handle_message("find office chairs")
    approved = agent.handle_message("find office chairs", approved_by_human=True)

    assert waiting.action == "request_human_review"
    assert waiting.requires_human_review is True
    assert approved.action == "search_products"
    assert service.queries[0]["query"] == "find office chairs"


def test_agent_fallback_for_unknown_intent():
    agent, service = build_agent()

    response = agent.handle_message("hello there")

    assert response.action == "fallback"
    assert "share a bit more detail" in response.content
    assert service.queries == []


def test_product_search_tool_rejects_empty_query():
    tool = ProductSearchTool(FakeSearchService())

    with pytest.raises(ValueError, match="message must not be empty"):
        tool.run("   ")


def test_format_product_hits_handles_empty_results():
    assert (
        format_product_hits([]) == "I could not find matching products in the catalog."
    )


def test_build_parser_parses_agent_options():
    args = build_parser().parse_args(
        [
            "show headphones",
            "--top-k",
            "4",
            "--approved-by-human",
            "--require-human-approval-for-tools",
        ]
    )

    assert args.message == "show headphones"
    assert args.top_k == 4
    assert args.approved_by_human is True
    assert args.require_human_approval_for_tools is True


def graph_input(content, approved_by_human=False):
    return {
        "messages": [{"role": "user", "content": content}],
        "approved_by_human": approved_by_human,
        "tool_outputs": [],
    }


class FakeStreamingLLM(FakeToolCallingLLM):
    """Adds a .stream() that yields the answer one word at a time."""

    def stream(self, messages):
        self.messages = messages
        self.invoke_count += 1
        saw_tool_result = any(
            "Result of " in str(m.get("content", "")) for m in messages
        )
        if not saw_tool_result:
            self.tool_call_count += 1
            yield SimpleNamespace(
                content="",
                tool_calls=[{"name": self.tool_name, "args": self.args}],
                usage_metadata={"input_tokens": 42, "output_tokens": 7},
                response_metadata={},
            )
            return
        for word in self.answer.split():
            yield SimpleNamespace(
                content=word + " ",
                tool_calls=[],
                usage_metadata={"input_tokens": 30, "output_tokens": 12},
                response_metadata={},
            )


def test_agent_streams_tokens_through_the_callback():
    llm = FakeStreamingLLM(
        "search_products",
        {"query": "headphones", "top_k": 1},
        answer="Found two great options",
    )
    agent = SmartShopAgent(
        config=AgentConfig(top_k=2, use_llm_routing="always"),
        search_tool=ProductSearchTool(FakeSearchService()),
        llm=llm,
    )

    tokens = []
    response = agent.handle_message("show me headphones", on_token=tokens.append)

    # Tokens arrive incrementally, not as one blob after the fact.
    assert len(tokens) == 4
    assert "".join(tokens).strip() == "Found two great options"
    assert response.content.strip() == "Found two great options"


def test_agent_falls_back_to_invoke_without_a_token_callback():
    llm = FakeStreamingLLM("search_products", {"query": "x", "top_k": 1})
    agent = SmartShopAgent(
        config=AgentConfig(top_k=2, use_llm_routing="always"),
        search_tool=ProductSearchTool(FakeSearchService()),
        llm=llm,
    )

    response = agent.handle_message("show me headphones")

    assert response.content


def test_supervisor_delegates_each_intent_to_its_specialist():
    from src.agent import SupervisorAgent

    core, _service = build_agent()
    supervisor = SupervisorAgent(core)

    cases = {
        "show me wireless headphones": "product_agent",
        "what is your shipping policy?": "policy_agent",
        "I want a refund and chargeback": "support_agent",
        "zzz": "fallback_agent",
    }
    for message, expected in cases.items():
        decision = supervisor.route(message)
        assert supervisor.specialist_for(decision.action).name == expected


def test_specialist_records_itself_in_the_trace():
    from src.agent import SupervisorAgent

    core, _service = build_agent()
    supervisor = SupervisorAgent(core)

    response = supervisor.handle_message("show me wireless headphones")

    delegated = [e for e in response.trace_events if e.event == "delegate"]
    assert delegated[0].metadata["agent"] == "product_agent"
    assert response.action == "search_products"


def test_langgraph_graph_exposes_one_node_per_specialist():
    pytest.importorskip("langgraph")
    agent, _service = build_agent()
    app = build_langgraph_app(agent)

    nodes = set(app.get_graph().nodes)
    assert {"supervisor", "product_agent", "policy_agent", "support_agent"} <= nodes


def test_graph_state_reports_the_handling_agent():
    pytest.importorskip("langgraph")
    agent, _service = build_agent()
    app = build_langgraph_app(agent)

    state = app.invoke(
        graph_input("show me wireless headphones"),
        config={"configurable": {"thread_id": "t-agent"}},
    )

    assert state["agent"] == "product_agent"


def test_langgraph_app_runs_when_dependency_is_available():
    pytest.importorskip("langgraph")
    agent, _service = build_agent()
    app = build_langgraph_app(agent)

    state = app.invoke(
        graph_input("what is your return policy?"),
        config={"configurable": {"thread_id": "t-policy"}},
    )

    assert state["next_action"] == "answer_policy"
    assert "30 days" in state["content"]


def test_langgraph_app_persists_state_per_thread():
    pytest.importorskip("langgraph")
    agent, _service = build_agent()
    app = build_langgraph_app(agent)
    config = {"configurable": {"thread_id": "t-memory"}}

    app.invoke(graph_input("what is your shipping policy?"), config=config)
    snapshot = app.get_state(config)

    # The checkpointer kept the first turn, which is what makes the graph
    # stateful rather than a fresh run per invocation.
    assert snapshot.values["messages"]
    assert snapshot.values["next_action"] == "answer_policy"

    app.invoke(graph_input("show me wireless headphones"), config=config)
    assert app.get_state(config).values["next_action"] == "search_products"

    # A different thread must not see the first thread's conversation.
    other = {"configurable": {"thread_id": "t-other"}}
    assert app.get_state(other).values == {}


def test_langgraph_app_interrupts_before_human_review():
    pytest.importorskip("langgraph")
    agent, _service = build_agent()
    app = build_langgraph_app(agent)
    config = {"configurable": {"thread_id": "t-review"}}

    app.invoke(graph_input("I want a refund and a chargeback"), config=config)
    snapshot = app.get_state(config)

    # The run halts *before* executing human review instead of answering and
    # trusting the client to come back.
    assert snapshot.next == ("request_human_review",)
    assert snapshot.values.get("content") is None

    # Resuming continues from the checkpoint rather than re-routing.
    resumed = app.invoke(None, config=config)
    assert resumed["next_action"] == "request_human_review"
    assert resumed["requires_human_review"] is True


def test_langgraph_app_routes_once_per_turn():
    pytest.importorskip("langgraph")
    service = FakeSearchService()
    llm = FakeToolCallingLLM("search_products", {"query": "desk lamp", "top_k": 1})
    agent = SmartShopAgent(
        config=AgentConfig(top_k=2, use_llm_routing="always"),
        search_tool=ProductSearchTool(service),
        llm=llm,
    )
    app = build_langgraph_app(agent, interrupt_before_human_review=False)

    app.invoke(
        graph_input("anything for my desk?"),
        config={"configurable": {"thread_id": "t-once"}},
    )

    # Routing happens once: the router node decides and the action node reuses
    # that decision. A second routing call would mean paying twice for the same
    # decision. (The extra invoke is the synthesis call over the tool result.)
    assert llm.tool_call_count == 1
    assert llm.invoke_count == 2
