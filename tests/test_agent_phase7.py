import pytest

from src.agent import (
    AgentConfig,
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


def test_langgraph_app_runs_when_dependency_is_available():
    pytest.importorskip("langgraph")
    agent, _service = build_agent()
    app = build_langgraph_app(agent)

    state = app.invoke(
        {
            "messages": [{"role": "user", "content": "what is your return policy?"}],
            "approved_by_human": False,
            "tool_outputs": [],
        }
    )

    assert state["next_action"] == "answer_policy"
    assert "30 days" in state["content"]
