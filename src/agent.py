"""Phase 7 customer-support agent framework for SmartShop."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, Sequence, TypedDict

from src.vector_store import ProductSearchFilters, VectorSearchService

logger = logging.getLogger(__name__)

AgentAction = Literal[
    "answer_policy",
    "search_products",
    "request_human_review",
    "handoff_to_human",
    "fallback",
]

DEFAULT_POLICY_SOURCE_PATH = "data/policies/smartshop_policy.md"
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
class PolicyDocument:
    topic: str
    content: str
    source: str
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AgentConfig:
    top_k: int = 3
    require_human_approval_for_tools: bool = False
    use_llm_routing: str = field(
        default_factory=lambda: os.getenv("SMARTSHOP_AGENT_USE_LLM", "auto")
    )
    llm_model: str = field(
        default_factory=lambda: os.getenv("SMARTSHOP_OPENAI_MODEL", "o4-mini")
    )
    policy_source_path: str = field(
        default_factory=lambda: os.getenv(
            "SMARTSHOP_POLICY_SOURCE_PATH",
            DEFAULT_POLICY_SOURCE_PATH,
        )
    )
    llm_timeout_seconds: float = 30.0
    allow_rule_based_fallback: bool = True
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
    tool_args: dict[str, Any] = field(default_factory=dict)
    llm_model: str | None = None


@dataclass(frozen=True)
class AgentTraceEvent:
    event: str
    action: str | None = None
    tool: str | None = None
    status: str = "ok"
    latency_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentResponse:
    content: str
    action: AgentAction
    messages: list[dict[str, Any]]
    tool_outputs: list[dict[str, Any]] = field(default_factory=list)
    requires_human_review: bool = False
    reason: str = ""
    approval_request_id: str | None = None
    trace_events: list[AgentTraceEvent] = field(default_factory=list)

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


def format_policy_answer(document: PolicyDocument | None) -> str:
    if document is None:
        return (
            "I could not find a matching policy source. A human support "
            "specialist should review this request."
        )
    return f"{document.content}\n\nSource: {document.source}"


class PolicySource:
    """Small file-backed policy retriever used as a local RAG source."""

    def __init__(
        self,
        path: str | Path = DEFAULT_POLICY_SOURCE_PATH,
        fallback_topics: dict[str, str] | None = None,
    ):
        self.path = Path(path)
        self.fallback_topics = fallback_topics or DEFAULT_POLICY_TOPICS
        self._documents: list[PolicyDocument] | None = None

    def _load_documents(self) -> list[PolicyDocument]:
        if self._documents is not None:
            return self._documents

        if self.path.exists():
            self._documents = self._parse_markdown(
                self.path.read_text(encoding="utf-8")
            )
        else:
            self._documents = [
                PolicyDocument(
                    topic=topic,
                    content=content,
                    source=f"fallback:{topic}",
                )
                for topic, content in self.fallback_topics.items()
            ]
        return self._documents

    def _parse_markdown(self, raw: str) -> list[PolicyDocument]:
        documents: list[PolicyDocument] = []
        current_topic: str | None = None
        current_lines: list[str] = []

        def flush() -> None:
            if current_topic is None:
                return
            content = "\n".join(line.strip() for line in current_lines).strip()
            if content:
                documents.append(
                    PolicyDocument(
                        topic=current_topic,
                        content=content,
                        source=f"{self.path}#{current_topic}",
                    )
                )

        for line in raw.splitlines():
            if line.startswith("## "):
                flush()
                current_topic = line[3:].strip().lower().replace(" ", "-")
                current_lines = []
            elif current_topic:
                current_lines.append(line)
        flush()

        if not documents:
            raise ValueError(f"No policy sections found in {self.path}.")
        return documents

    def get(self, topic: str | None) -> PolicyDocument | None:
        if not topic:
            return None
        normalized = topic.strip().lower().replace(" ", "-")
        for document in self._load_documents():
            if document.topic == normalized:
                return document
        return None

    def search(self, query: str, top_k: int = 1) -> list[PolicyDocument]:
        text = _normalize_text(query)
        query_terms = {term for term in text.replace("-", " ").split() if len(term) > 2}
        scored: list[PolicyDocument] = []
        for document in self._load_documents():
            haystack = f"{document.topic} {document.content}".lower()
            keyword_score = 0
            for topic, keywords in POLICY_KEYWORDS.items():
                if topic == document.topic and _contains_any(text, keywords):
                    keyword_score += 5
            overlap_score = sum(1 for term in query_terms if term in haystack)
            score = float(keyword_score + overlap_score)
            if score > 0:
                scored.append(
                    PolicyDocument(
                        topic=document.topic,
                        content=document.content,
                        source=document.source,
                        score=score,
                    )
                )
        scored.sort(key=lambda doc: doc.score, reverse=True)
        return scored[:top_k]


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
        policy_source: PolicySource | None = None,
        llm: Any | None = None,
    ):
        self.config = config or AgentConfig()
        self.search_tool = search_tool or ProductSearchTool()
        self.policy_source = policy_source or PolicySource(
            self.config.policy_source_path,
            fallback_topics=self.config.policy_topics,
        )
        self.llm = llm or self._build_llm()
        self.llm_with_tools = self._bind_tools(self.llm) if self.llm else None

    def _should_use_llm(self) -> bool:
        mode = self.config.use_llm_routing.strip().lower()
        if mode in {"0", "false", "no", "never", "off"}:
            return False
        if mode in {"1", "true", "yes", "always", "on"}:
            return True
        return bool(os.getenv("OPENAI_API_KEY"))

    def _build_llm(self) -> Any | None:
        if not self._should_use_llm():
            return None
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            if self.config.allow_rule_based_fallback:
                logger.warning("langchain-openai unavailable; using fallback router.")
                return None
            raise RuntimeError(
                "langchain-openai is required for LLM routing. "
                "Install dependencies and set OPENAI_API_KEY."
            ) from exc

        return ChatOpenAI(
            model=self.config.llm_model,
            timeout=self.config.llm_timeout_seconds,
        )

    def _tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_products",
                    "description": "Search the SmartShop product catalog.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Customer product search query.",
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Maximum number of products to return.",
                                "minimum": 1,
                                "maximum": 10,
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "retrieve_policy",
                    "description": "Retrieve a SmartShop policy from the policy source.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "topic": {
                                "type": "string",
                                "description": "Policy topic such as shipping, return, payment, or warranty.",
                            },
                            "query": {
                                "type": "string",
                                "description": "Customer question to retrieve policy context for.",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "request_human_review",
                    "description": (
                        "Queue sensitive, risky, or account-specific requests "
                        "for human support approval."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {
                                "type": "string",
                                "description": "Why this needs human review.",
                            }
                        },
                        "required": ["reason"],
                    },
                },
            },
        ]

    def _bind_tools(self, llm: Any) -> Any:
        if llm is None:
            return None
        if not hasattr(llm, "bind_tools"):
            return llm
        return llm.bind_tools(self._tool_schemas())

    def _conversation_for_llm(
        self,
        message: str,
        history: Sequence[dict[str, Any]] | None,
        approved_by_human: bool,
    ) -> list[dict[str, str]]:
        policy_topics = ", ".join(
            document.topic for document in self.policy_source.search(message, top_k=4)
        )
        if not policy_topics:
            policy_topics = "shipping, return, payment, warranty"
        system = (
            "You are SmartShop's customer-support agent. Choose exactly one tool "
            "when a tool is useful: search_products for catalog searches, "
            "retrieve_policy for policy questions, or request_human_review for "
            "refund disputes, damaged/lost orders, legal/privacy/payment-risk, "
            "or anything requiring account-specific approval. "
            f"Human approval already present: {approved_by_human}. "
            f"Available policy topics from source: {policy_topics}."
        )
        messages = [{"role": "system", "content": system}]
        for row in history or []:
            role = str(row.get("role", "user"))
            content = str(row.get("content", ""))
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": message})
        return messages

    def _first_tool_call(self, ai_message: Any) -> tuple[str | None, dict[str, Any]]:
        tool_calls = getattr(ai_message, "tool_calls", None) or []
        if not tool_calls:
            raw_calls = getattr(ai_message, "additional_kwargs", {}).get(
                "tool_calls", []
            )
            tool_calls = raw_calls or []
        if not tool_calls:
            return None, {}

        call = tool_calls[0]
        if isinstance(call, dict):
            name = call.get("name") or call.get("function", {}).get("name")
            args = call.get("args") or call.get("function", {}).get("arguments") or {}
        else:
            name = getattr(call, "name", None)
            args = getattr(call, "args", {}) or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        return name, dict(args)

    def _llm_decide(
        self,
        message: str,
        history: Sequence[dict[str, Any]] | None = None,
        approved_by_human: bool = False,
    ) -> AgentDecision | None:
        if self.llm_with_tools is None:
            return None

        start = time.perf_counter()
        try:
            ai_message = self.llm_with_tools.invoke(
                self._conversation_for_llm(message, history, approved_by_human)
            )
        except Exception as exc:  # noqa: BLE001
            if self.config.allow_rule_based_fallback:
                logger.warning("LLM routing failed; using fallback router: %s", exc)
                return None
            raise

        elapsed = time.perf_counter() - start
        tool_name, args = self._first_tool_call(ai_message)
        logger.info(
            "agent_llm_decision",
            extra={
                "tool": tool_name or "none",
                "latency_seconds": elapsed,
                "model": self.config.llm_model,
            },
        )

        if tool_name == "request_human_review" and not approved_by_human:
            return AgentDecision(
                action="request_human_review",
                reason=str(args.get("reason") or "LLM requested human review."),
                tool_args=args,
                llm_model=self.config.llm_model,
            )
        if tool_name == "search_products":
            if self.config.require_human_approval_for_tools and not approved_by_human:
                return AgentDecision(
                    action="request_human_review",
                    reason="Tool execution requires human approval.",
                    tool_args=args,
                    llm_model=self.config.llm_model,
                )
            return AgentDecision(
                action="search_products",
                reason="LLM selected search_products tool.",
                tool_args=args,
                llm_model=self.config.llm_model,
            )
        if tool_name == "retrieve_policy":
            topic = str(args.get("topic") or "").strip().lower().replace(" ", "-")
            if not topic:
                matches = self.policy_source.search(str(args.get("query") or message))
                topic = matches[0].topic if matches else None
            return AgentDecision(
                action="answer_policy",
                reason="LLM selected retrieve_policy tool.",
                policy_topic=topic,
                tool_args=args,
                llm_model=self.config.llm_model,
            )
        if approved_by_human:
            return AgentDecision(
                action="handoff_to_human",
                reason="Human approval is present for a complex request.",
                llm_model=self.config.llm_model,
            )
        return AgentDecision(
            action="fallback",
            reason="LLM did not select a supported tool.",
            llm_model=self.config.llm_model,
        )

    def decide(self, message: str, approved_by_human: bool = False) -> AgentDecision:
        return self.decide_with_history(message, None, approved_by_human)

    def decide_with_history(
        self,
        message: str,
        history: Sequence[dict[str, Any]] | None = None,
        approved_by_human: bool = False,
    ) -> AgentDecision:
        llm_decision = self._llm_decide(
            message,
            history=history,
            approved_by_human=approved_by_human,
        )
        if llm_decision is not None:
            return llm_decision

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
                    reason=f"Fallback matched policy topic: {topic}.",
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
                reason="Fallback matched product-search intent.",
            )

        if approved_by_human:
            return AgentDecision(
                action="handoff_to_human",
                reason="Human approval is present for a complex request.",
            )

        return AgentDecision(
            action="fallback",
            reason="No LLM/tool, policy, or product-search intent matched.",
        )

    def handle_message(
        self,
        message: str,
        history: Sequence[dict[str, Any]] | None = None,
        approved_by_human: bool = False,
    ) -> AgentResponse:
        user_message = {"role": "user", "content": message}
        messages = list(history or []) + [user_message]
        trace_events: list[AgentTraceEvent] = []
        decision = self.decide_with_history(
            message,
            history=history,
            approved_by_human=approved_by_human,
        )
        trace_events.append(
            AgentTraceEvent(
                event="decision",
                action=decision.action,
                metadata={
                    "reason": decision.reason,
                    "llm_model": decision.llm_model,
                },
            )
        )
        logger.info(
            "agent_decision",
            extra={"action": decision.action, "reason": decision.reason},
        )

        if decision.action == "answer_policy":
            document = self.policy_source.get(decision.policy_topic)
            if document is None:
                matches = self.policy_source.search(
                    str(decision.tool_args.get("query") or message)
                )
                document = matches[0] if matches else None
            content = format_policy_answer(document)
            messages.append({"role": "assistant", "content": content})
            return AgentResponse(
                content=content,
                action=decision.action,
                messages=messages,
                reason=decision.reason,
                trace_events=trace_events,
            )

        if decision.action == "search_products":
            query = str(decision.tool_args.get("query") or message)
            top_k = int(decision.tool_args.get("top_k") or self.config.top_k)
            start = time.perf_counter()
            hits = self.search_tool.run(query, top_k=top_k)
            trace_events.append(
                AgentTraceEvent(
                    event="tool_call",
                    action=decision.action,
                    tool="search_products",
                    latency_seconds=time.perf_counter() - start,
                    metadata={"top_k": top_k},
                )
            )
            content = format_product_hits(hits)
            tool_output = {"tool": "search_products", "query": query, "hits": hits}
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
                trace_events=trace_events,
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
                trace_events=trace_events,
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
                trace_events=trace_events,
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
            trace_events=trace_events,
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
