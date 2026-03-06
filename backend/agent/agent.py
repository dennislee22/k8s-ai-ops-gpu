"""
LangGraph Agent — K8s AI Ops Assistant
Reason → Act → Observe → Synthesize loop using local Ollama LLM.
Phase 1: K8s tools only
Phase 2: K8s tools + RAG document retrieval
"""

import sys as _sys
from pathlib import Path as _Path
_BACKEND_DIR = _Path(__file__).resolve().parent
while not (_BACKEND_DIR / "main.py").exists() and _BACKEND_DIR != _BACKEND_DIR.parent:
    _BACKEND_DIR = _BACKEND_DIR.parent
if str(_BACKEND_DIR) not in _sys.path:
    _sys.path.insert(0, str(_BACKEND_DIR))

import os
import json
import time
from typing import Annotated, TypedDict, Literal
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

load_dotenv("env")

from core.logger import get_logger
logger = get_logger(__name__)

from tools.k8s_tool import K8S_TOOLS

PHASE           = int(os.getenv("PHASE", "2"))
LLM_MODEL       = os.getenv("LLM_MODEL", "qwen2.5:7b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

NUM_THREAD = int(os.getenv("NUM_THREAD", "16"))
NUM_CTX    = int(os.getenv("NUM_CTX", "4096"))
NUM_GPU    = int(os.getenv("NUM_GPU", "0"))

# ── System prompt ─────────────────────────────────────────────────────────
# Edit CUSTOM_RULES below to add your own site-specific instructions.
# These are injected into every LLM call alongside the base rules.

CUSTOM_RULES = os.getenv("CUSTOM_RULES", """
- Do NOT recommend migrating to cgroupv2. This environment uses cgroupv1 and migration is not supported.
- Do NOT suggest changes to the container runtime. The runtime is managed separately.
""").strip()

SYSTEM_PROMPT = """You are an expert Kubernetes operations assistant running in an air-gapped environment.
Your job is to diagnose cluster issues, check system health, and provide actionable recommendations.

LANGUAGE RULE:
- ALWAYS respond in English only. Never use any other language regardless of how the user writes.

ENVIRONMENT CONTEXT:
- This is a production Kubernetes cluster.
- Longhorn is the distributed block storage solution used for persistent volumes. It runs as pods
  in the longhorn-system namespace. When diagnosing storage issues, check Longhorn manager,
  engine, and replica pods. Common Longhorn issues include: replica rebuilding, volume degraded,
  node disk pressure, and engine image upgrades.
- Storage classes: 'longhorn' is the default storage class backed by Longhorn.

CRITICAL RULES:
1. NEVER fabricate data. Only state what the tools actually returned.
2. Always cite which tool result you are reasoning from.
3. Be specific — name the exact pod, node, or deployment that has the issue.
4. Rank your diagnosis: list the most likely cause first with evidence.
5. NEVER suggest write operations (restart, delete, scale). Only diagnose and recommend.
6. When asked about cluster health, ALWAYS scan ALL namespaces, not just 'default'.
   Call tools with namespace='all' or iterate across key namespaces.
7. For storage issues, ALWAYS check the longhorn-system namespace for Longhorn pod health.
{rag_instruction}

SITE-SPECIFIC RULES (do not override these):
{custom_rules}

RESPONSE FORMAT:
- Be concise. Use short bullet points. No lengthy paragraphs.
- State what you found, what it means, and what to do — nothing more.
- Do NOT add disclaimers, caveats, or generic advice unrelated to the actual findings.
- Omit sections that have nothing to report (e.g. skip "Node Health" if all nodes are fine).
- Max response length: ~300 words unless the issue is genuinely complex.
"""

RAG_INSTRUCTION = """
8. ALWAYS search documentation before finalizing a diagnosis. If you observe an anomaly,
   search for matching known issues. Cross-reference live data with documentation.
9. When a known issue matches, cite the document source and its recommended fix.
"""


# ── Build LangChain tools from registry ──────────────────────────────────

def _make_lc_tool(name: str, config: dict):
    fn   = config["fn"]
    desc = config["description"]
    params = config.get("parameters", {})

    if not params:
        @tool(name, description=desc)
        def _tool_no_params() -> str:
            return fn()
        return _tool_no_params
    else:
        param_desc = ", ".join(
            f"{k}: {v.get('type','str')} (default={v.get('default','required')})"
            for k, v in params.items()
        )
        full_desc = f"{desc}\nParameters: {param_desc}"

        @tool(name, description=full_desc)
        def _tool_with_params(tool_input: str) -> str:
            """Execute tool with JSON input."""
            try:
                kwargs = json.loads(tool_input) if tool_input.strip().startswith("{") else {}
            except json.JSONDecodeError:
                kwargs = {}
            for k, v in params.items():
                if k not in kwargs and "default" in v:
                    kwargs[k] = v["default"]
            return fn(**kwargs)

        return _tool_with_params


def build_tools():
    tools = []
    for name, config in K8S_TOOLS.items():
        tools.append(_make_lc_tool(name, config))
        logger.debug(f"[Tools] Registered K8s tool: {name}")

    if PHASE >= 2:
        from rag.rag_tool import RAG_TOOLS
        for name, config in RAG_TOOLS.items():
            tools.append(_make_lc_tool(name, config))
            logger.debug(f"[Tools] Registered RAG tool: {name}")
        logger.info(f"[Tools] Phase 2 — {len(tools)} tools loaded (K8s + RAG)")
    else:
        logger.info(f"[Tools] Phase 1 — {len(tools)} tools loaded (K8s only)")

    return tools


# ── Agent state ───────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    tool_calls_made: list[str]
    iteration: int
    status_updates: list[str]   # real-time status messages streamed to frontend


# ── Build agent ───────────────────────────────────────────────────────────

def build_agent():
    tools    = build_tools()
    tool_map = {t.name: t for t in tools}

    system_prompt = SYSTEM_PROMPT.format(
        rag_instruction=RAG_INSTRUCTION if PHASE >= 2 else "",
        custom_rules=CUSTOM_RULES,
    )

    logger.info(
        f"[LLM] Initialising ChatOllama — model={LLM_MODEL} "
        f"ctx={NUM_CTX} threads={NUM_THREAD} gpu={NUM_GPU}"
    )

    llm = ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,
        num_ctx=NUM_CTX,
        num_thread=NUM_THREAD,
        num_gpu=NUM_GPU,
        repeat_penalty=1.1,
    ).bind_tools(tools)

    logger.info("[LLM] ChatOllama ready")

    # ── LLM node ──────────────────────────────────────────────────────────
    def llm_node(state: AgentState) -> AgentState:
        iteration = state.get("iteration", 0) + 1
        updates   = list(state.get("status_updates", []))

        msg = (
            f"🧠 Reasoning... (iteration {iteration})"
            if iteration == 1
            else f"🔄 Synthesising findings... (iteration {iteration})"
        )
        updates.append(msg)
        logger.info(f"[Agent] LLM iteration {iteration}")

        t0       = time.time()
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = llm.invoke(messages)
        elapsed  = time.time() - t0

        tool_calls = getattr(response, "tool_calls", [])
        if tool_calls:
            names = [tc['name'] for tc in tool_calls]
            updates.append(f"🔧 Calling tools: {', '.join(names)}")

        logger.info(
            f"[Agent] LLM responded in {elapsed:.1f}s — "
            f"tool_calls={[tc['name'] for tc in tool_calls]}"
        )
        return {
            "messages": [response],
            "tool_calls_made": state.get("tool_calls_made", []),
            "iteration": iteration,
            "status_updates": updates,
        }

    # ── Tool node ─────────────────────────────────────────────────────────
    def tool_node(state: AgentState) -> AgentState:
        last_message = state["messages"][-1]
        tool_results = []
        tools_called = list(state.get("tool_calls_made", []))
        updates      = list(state.get("status_updates", []))

        # Human-friendly tool descriptions for status messages
        tool_labels = {
            "get_pod_status":        "📦 Checking pod status",
            "get_node_health":       "🖥️  Checking node health",
            "get_pod_logs":          "📋 Fetching pod logs",
            "get_events":            "⚠️  Fetching cluster events",
            "get_deployment_status": "🚀 Checking deployments",
            "describe_pod":          "🔍 Describing pod",
            "search_documentation":  "📚 Searching knowledge base",
        }

        for tc in last_message.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            tools_called.append(tool_name)

            label = tool_labels.get(tool_name, f"⚙️  Running {tool_name}")
            ns    = tool_args.get("namespace", "")
            if ns and ns != "default":
                label += f" ({ns})"
            updates.append(label)

            logger.info(f"[Tool] Calling '{tool_name}' args={tool_args}")
            t0 = time.time()
            try:
                tool_fn = tool_map.get(tool_name)
                if not tool_fn:
                    result = f"Error: Tool '{tool_name}' not found."
                    logger.error(f"[Tool] '{tool_name}' not in registry")
                else:
                    result = tool_fn.invoke(json.dumps(tool_args) if tool_args else {})
                elapsed = time.time() - t0
                preview = str(result)[:120].replace("\n", " ")
                logger.info(f"[Tool] '{tool_name}' returned in {elapsed:.2f}s — {preview}...")
            except Exception as e:
                result = f"Tool '{tool_name}' failed: {str(e)}"
                logger.error(f"[Tool] '{tool_name}' exception: {e}", exc_info=True)

            tool_results.append(
                ToolMessage(content=str(result), tool_call_id=tc["id"])
            )

        return {
            "messages": tool_results,
            "tool_calls_made": tools_called,
            "iteration": state.get("iteration", 0),
            "status_updates": updates,
        }

    # ── Router ────────────────────────────────────────────────────────────
    def should_continue(state: AgentState) -> Literal["tools", "end"]:
        last      = state["messages"][-1]
        iteration = state.get("iteration", 0)
        if iteration >= 8:
            logger.warning("[Agent] Max iterations (8) reached — forcing end")
            return "end"
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        logger.info(f"[Agent] Finished after {iteration} iteration(s)")
        return "end"

    graph = StateGraph(AgentState)
    graph.add_node("llm",   llm_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("llm")
    graph.add_conditional_edges("llm", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "llm")

    logger.info("[Agent] LangGraph compiled and ready")
    return graph.compile()


# ── Public interface ──────────────────────────────────────────────────────

_agent = None

def get_agent():
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent


async def run_agent(user_message: str) -> dict:
    """Run agent. Scans all namespaces — no namespace filtering at this level."""
    agent = get_agent()
    logger.info(f"[Agent] Running query: {user_message[:80]}...")

    initial_state = {
        "messages": [HumanMessage(content=user_message)],
        "tool_calls_made": [],
        "iteration": 0,
        "status_updates": [
            f"🤖 Model: {LLM_MODEL}",
        ],
    }

    t0          = time.time()
    final_state = await agent.ainvoke(initial_state)
    elapsed     = time.time() - t0

    final_message = final_state["messages"][-1]
    response_text = final_message.content if hasattr(final_message, "content") else str(final_message)

    tools_used     = final_state.get("tool_calls_made", [])
    iterations     = final_state.get("iteration", 0)
    status_updates = final_state.get("status_updates", [])
    status_updates.append(f"✅ Done in {elapsed:.0f}s")

    logger.info(
        f"[Agent] Complete — {elapsed:.1f}s | "
        f"iterations={iterations} | tools={tools_used}"
    )

    return {
        "response":       response_text,
        "tools_used":     tools_used,
        "iterations":     iterations,
        "phase":          PHASE,
        "status_updates": status_updates,
        "elapsed_seconds": round(elapsed, 1),
    }
