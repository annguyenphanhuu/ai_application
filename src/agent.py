"""Phase 7 customer-support agent framework for SmartShop."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol, Sequence, TypedDict

from src.vector_store import ProductSearchFilters, VectorSearchService


AgentAction = Literal[
    "answer_policy",
    "search_products",
    "request_human_review",
    "handoff_to_human",
    "fallback",
]

DEFAULT_POLICY_TOPICS = {
    "shipping": (
        "Standard shipping takes 3-5 business days. Express shipping takes "
        "1-2 business days when available."
    ),
    "return": (
        "Most products can be returned within 30 days if they are unused and "
        "in the original packaging."
    ),
    "payment": (
        "SmartShop supports major cards, bank transfer, and supported digital "
        "wallets. Payment details should never be shared in chat."
    ),
    "warranty": (
        "Warranty coverage depends on the product and brand. Share the product "
        "ID or order ID so the support team can verify exact coverage."
    ),
}

PRODUCT_INTENT_KEYWORDS = (
    "buy",
    "find",
    "need",
    "recommend",
    "search",
    "show",
    "looking for",
    "product",
    "headphone",
    "chair",
    "laptop",
    "phone",
    "camera",
    "mua",
    "tim",
    "tim kiem",
    "goi y",
    "san pham",
)

HUMAN_REVIEW_KEYWORDS = (
    "refund",
    "chargeback",
    "cancel order",
    "lost package",
    "damaged",
    "complaint",
    "legal",
    "lawsuit",
    "password",
    "credit card",
    "personal data",
    "hoan tien",
    "khieu nai",
    "huy don",
    "mat hang",
)

POLICY_KEYWORDS = {
    "shipping": ("shipping", "delivery", "ship", "giao hang", "van chuyen"),
    "return": ("return", "exchange", "refund policy", "doi tra", "tra hang"),
    "payment": ("payment", "pay", "checkout", "thanh toan"),
    "warranty": ("warranty", "guarantee", "bao hanh"),
}


class SearchService(Protocol):
    def search_products(
        self,
        query: str,
        filters: ProductSearchFilters | None = None,
        top_k: int = 5,
        category_filter: str | None = None,
    ) -> list[dict]:
        """Search products and return normalized product dictionaries."""


class AgentState(TypedDict, total=False):
    messages: list[dict[str, Any]]
    next_action: AgentAction
    approved_by_human: bool
    tool_outputs: list[dict[str, Any]]
    reason: str
    content: str
    requires_human_review: bool


@dataclass(frozen=True)
class AgentConfig:
    top_k: int = 3
    require_human_approval_for_tools: bool = False
    policy_topics: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_POLICY_TOPICS)
    )

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be greater than zero.")


@dataclass(frozen=True)
class AgentDecision:
    action: AgentAction
    reason: str
    policy_topic: str | None = None


@dataclass(frozen=True)
class AgentResponse:
    content: str
    action: AgentAction
    messages: list[dict[str, Any]]
    tool_outputs: list[dict[str, Any]] = field(default_factory=list)
    requires_human_review: bool = False
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_text(value: str) -> str:
    normalized = " ".join(value.strip().lower().split())
    if not normalized:
        raise ValueError("message must not be empty.")
    return normalized


def _contains_any(text: str, keywords: Sequence[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def format_product_hits(hits: Sequence[dict]) -> str:
    if not hits:
        return "I could not find matching products in the catalog."

    lines = ["Here are matching products:"]
    for index, hit in enumerate(hits, start=1):
        product_id = hit.get("product_id") or "unknown"
        title = hit.get("title") or "Untitled product"
        price = hit.get("price")
        score = hit.get("score")

        details = [f"{index}. {title}", f"id={product_id}"]
        if price is not None:
            details.append(f"price={price}")
        if score is not None:
            details.append(f"score={float(score):.2f}")
        lines.append(" | ".join(details))
    return "\n".join(lines)


class ProductSearchTool:
    def __init__(self, search_service: SearchService | None = None):
        self.search_service = search_service

    def _service(self) -> SearchService:
        if self.search_service is None:
            self.search_service = VectorSearchService()
        return self.search_service

    def run(
        self,
        query: str,
        top_k: int = 3,
        filters: ProductSearchFilters | None = None,
    ) -> list[dict]:
        _normalize_text(query)
        return self._service().search_products(query, filters=filters, top_k=top_k)


class SmartShopAgent:
    def __init__(
        self,
        config: AgentConfig | None = None,
        search_tool: ProductSearchTool | None = None,
    ):
        self.config = config or AgentConfig()
        self.search_tool = search_tool or ProductSearchTool()

    def decide(self, message: str, approved_by_human: bool = False) -> AgentDecision:
        text = _normalize_text(message)

        if _contains_any(text, HUMAN_REVIEW_KEYWORDS) and not approved_by_human:
            return AgentDecision(
                action="request_human_review",
                reason="Message contains a sensitive support intent.",
            )

        for topic, keywords in POLICY_KEYWORDS.items():
            if _contains_any(text, keywords):
                return AgentDecision(
                    action="answer_policy",
                    reason=f"Matched policy topic: {topic}.",
                    policy_topic=topic,
                )

        if _contains_any(text, PRODUCT_INTENT_KEYWORDS):
            if self.config.require_human_approval_for_tools and not approved_by_human:
                return AgentDecision(
                    action="request_human_review",
                    reason="Tool execution requires human approval.",
                )
            return AgentDecision(
                action="search_products",
                reason="Matched product-search intent.",
            )

        if approved_by_human:
            return AgentDecision(
                action="handoff_to_human",
                reason="Human approval is present for a complex request.",
            )

        return AgentDecision(
            action="fallback",
            reason="No policy or product-search intent matched.",
        )

    def handle_message(
        self,
        message: str,
        history: Sequence[dict[str, Any]] | None = None,
        approved_by_human: bool = False,
    ) -> AgentResponse:
        user_message = {"role": "user", "content": message}
        messages = list(history or []) + [user_message]
        decision = self.decide(message, approved_by_human=approved_by_human)

        if decision.action == "answer_policy":
            content = self.config.policy_topics[decision.policy_topic or ""]
            messages.append({"role": "assistant", "content": content})
            return AgentResponse(
                content=content,
                action=decision.action,
                messages=messages,
                reason=decision.reason,
            )

        if decision.action == "search_products":
            hits = self.search_tool.run(message, top_k=self.config.top_k)
            content = format_product_hits(hits)
            tool_output = {"tool": "search_products", "query": message, "hits": hits}
            messages.extend(
                [
                    {"role": "tool", "name": "search_products", "content": hits},
                    {"role": "assistant", "content": content},
                ]
            )
            return AgentResponse(
                content=content,
                action=decision.action,
                messages=messages,
                tool_outputs=[tool_output],
                reason=decision.reason,
            )

        if decision.action == "request_human_review":
            content = (
                "This request needs a human support review before I continue. "
                "I have paused the workflow and marked it for handoff."
            )
            messages.append({"role": "assistant", "content": content})
            return AgentResponse(
                content=content,
                action=decision.action,
                messages=messages,
                requires_human_review=True,
                reason=decision.reason,
            )

        if decision.action == "handoff_to_human":
            content = (
                "A human support specialist can now continue this case with the "
                "conversation context."
            )
            messages.append({"role": "assistant", "content": content})
            return AgentResponse(
                content=content,
                action=decision.action,
                messages=messages,
                reason=decision.reason,
            )

        content = (
            "I can help search products, explain shopping policies, or route a "
            "complex case to human support. Could you share a bit more detail?"
        )
        messages.append({"role": "assistant", "content": content})
        return AgentResponse(
            content=content,
            action=decision.action,
            messages=messages,
            reason=decision.reason,
        )

    def graph_state(
        self,
        message: str,
        history: Sequence[dict[str, Any]] | None = None,
        approved_by_human: bool = False,
    ) -> AgentState:
        response = self.handle_message(
            message,
            history=history,
            approved_by_human=approved_by_human,
        )
        return {
            "messages": response.messages,
            "next_action": response.action,
            "approved_by_human": approved_by_human,
            "tool_outputs": response.tool_outputs,
            "reason": response.reason,
        }


def build_langgraph_app(agent: SmartShopAgent | None = None) -> Any:
    try:
        from langgraph.graph import END, StateGraph
    except ImportError as exc:
        raise RuntimeError(
            "langgraph is required to compile the Phase 7 workflow. "
            "Install it with `pip install -r requirements.txt` or update environment.yml."
        ) from exc

    agent = agent or SmartShopAgent()

    def route_intent(state: AgentState) -> AgentState:
        messages = state.get("messages", [])
        if not messages:
            raise ValueError("AgentState.messages must contain at least one message.")
        last_content = str(messages[-1].get("content", ""))
        decision = agent.decide(
            last_content,
            approved_by_human=bool(state.get("approved_by_human", False)),
        )
        return {
            **state,
            "next_action": decision.action,
            "reason": decision.reason,
        }

    def answer_policy(state: AgentState) -> AgentState:
        last_content = str(state["messages"][-1].get("content", ""))
        response = agent.handle_message(
            last_content,
            history=state["messages"][:-1],
            approved_by_human=bool(state.get("approved_by_human", False)),
        )
        return {**state, **response.to_dict(), "next_action": response.action}

    def search_products(state: AgentState) -> AgentState:
        return answer_policy(state)

    def request_human_review(state: AgentState) -> AgentState:
        return answer_policy(state)

    def fallback(state: AgentState) -> AgentState:
        return answer_policy(state)

    def choose_node(state: AgentState) -> str:
        return str(state.get("next_action", "fallback"))

    workflow = StateGraph(AgentState)
    workflow.add_node("route_intent", route_intent)
    workflow.add_node("answer_policy", answer_policy)
    workflow.add_node("search_products", search_products)
    workflow.add_node("request_human_review", request_human_review)
    workflow.add_node("handoff_to_human", request_human_review)
    workflow.add_node("fallback", fallback)
    workflow.set_entry_point("route_intent")
    workflow.add_conditional_edges(
        "route_intent",
        choose_node,
        {
            "answer_policy": "answer_policy",
            "search_products": "search_products",
            "request_human_review": "request_human_review",
            "handoff_to_human": "handoff_to_human",
            "fallback": "fallback",
        },
    )
    workflow.add_edge("answer_policy", END)
    workflow.add_edge("search_products", END)
    workflow.add_edge("request_human_review", END)
    workflow.add_edge("handoff_to_human", END)
    workflow.add_edge("fallback", END)
    return workflow.compile()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SmartShop Phase 7 AI agent.")
    parser.add_argument("message", help="Customer message to handle.")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--approved-by-human", action="store_true")
    parser.add_argument(
        "--require-human-approval-for-tools",
        action="store_true",
        help="Pause before product-search tool execution.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    agent = SmartShopAgent(
        config=AgentConfig(
            top_k=args.top_k,
            require_human_approval_for_tools=args.require_human_approval_for_tools,
        )
    )
    response = agent.handle_message(
        args.message,
        approved_by_human=args.approved_by_human,
    )
    print(json.dumps(response.to_dict(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    main()
