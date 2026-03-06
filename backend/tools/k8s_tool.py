"""
Phase 1 — Kubernetes Tool
Read-only access to cluster state: pods, nodes, logs, events.
No write operations. All actions via Python kubernetes client.
"""

import sys as _sys
from pathlib import Path as _Path
_BACKEND_DIR = _Path(__file__).resolve().parent
while not (_BACKEND_DIR / "main.py").exists() and _BACKEND_DIR != _BACKEND_DIR.parent:
    _BACKEND_DIR = _BACKEND_DIR.parent
if str(_BACKEND_DIR) not in _sys.path:
    _sys.path.insert(0, str(_BACKEND_DIR))


import os
import time
from kubernetes import client, config
from kubernetes.client.rest import ApiException
from dotenv import load_dotenv

load_dotenv("env")

from core.logger import get_logger
logger = get_logger(__name__)


def _load_k8s_config():
    kubeconfig = os.getenv("KUBECONFIG_PATH", "")
    try:
        if kubeconfig and os.path.exists(os.path.expanduser(kubeconfig)):
            config.load_kube_config(config_file=os.path.expanduser(kubeconfig))
            logger.info(f"[K8s] Loaded kubeconfig from {kubeconfig}")
        else:
            config.load_incluster_config()
            logger.info("[K8s] Loaded in-cluster kubeconfig")
    except Exception as e:
        logger.error(f"[K8s] Failed to load config: {e}")
        raise RuntimeError(f"Failed to load K8s config: {e}")


_load_k8s_config()
_core = client.CoreV1Api()
_apps = client.AppsV1Api()


def _timed(fn_name: str):
    """Decorator factory to log execution time for each K8s call."""
    def decorator(fn):
        def wrapper(*args, **kwargs):
            t0 = time.time()
            logger.info(f"[K8s] → {fn_name}({args}, {kwargs})")
            try:
                result = fn(*args, **kwargs)
                logger.info(f"[K8s] ← {fn_name} completed in {time.time()-t0:.2f}s")
                return result
            except ApiException as e:
                logger.error(f"[K8s] ← {fn_name} API error {e.status}: {e.reason}")
                raise
            except Exception as e:
                logger.error(f"[K8s] ← {fn_name} error: {e}")
                raise
        return wrapper
    return decorator


@_timed("get_pod_status")
def get_pod_status(namespace: str = "default") -> str:
    try:
        pods = _core.list_pod_for_all_namespaces() if namespace == "all" else _core.list_namespaced_pod(namespace=namespace)
        if not pods.items:
            return f"No pods found in namespace '{namespace}'."
        lines = [f"Pods in namespace '{namespace}':"]
        skipped = 0
        for pod in pods.items:
            phase    = pod.status.phase or "Unknown"
            restarts = sum(cs.restart_count for cs in (pod.status.container_statuses or []))
            ready    = sum(1 for cs in (pod.status.container_statuses or []) if cs.ready)
            total    = len(pod.spec.containers)
            # Skip Completed/Succeeded pods that are fully ready — they are not issues
            if phase in ("Succeeded", "Completed"):
                skipped += 1
                continue
            # Also skip Running pods that are fully ready with 0 restarts — they are healthy
            if phase == "Running" and ready == total and restarts == 0:
                skipped += 1
                continue
            bad_cond = [f"{c.type}={c.status}" for c in (pod.status.conditions or []) if c.status != "True"]
            cond_str = f" [{', '.join(bad_cond)}]" if bad_cond else ""
            lines.append(
                f"  {pod.metadata.name}: {phase} | Ready {ready}/{total} | Restarts: {restarts}{cond_str}"
            )
        if skipped:
            lines.append(f"  ({skipped} healthy/completed pods omitted)")
        if len(lines) == 1:
            return f"All pods healthy in namespace '{namespace}' (no issues found)."
        return "\n".join(lines)
    except ApiException as e:
        return f"K8s API error: {e.reason}"


@_timed("get_node_health")
def get_node_health() -> str:
    try:
        nodes = _core.list_node()
        if not nodes.items:
            return "No nodes found."
        lines = ["Node health:"]
        for node in nodes.items:
            roles      = [k.replace("node-role.kubernetes.io/", "") for k in (node.metadata.labels or {}) if k.startswith("node-role.kubernetes.io/")] or ["worker"]
            conditions = {c.type: c.status for c in (node.status.conditions or [])}
            ready      = conditions.get("Ready", "Unknown")
            pressure   = [t for t in ["MemoryPressure","DiskPressure","PIDPressure"] if conditions.get(t) == "True"]
            alloc      = node.status.allocatable or {}
            lines.append(
                f"  {node.metadata.name} [{','.join(roles)}]: Ready={ready}"
                + (f" ⚠ {','.join(pressure)}" if pressure else "")
                + f" | CPU: {alloc.get('cpu','n/a')} | Mem: {alloc.get('memory','n/a')}"
            )
        return "\n".join(lines)
    except ApiException as e:
        return f"K8s API error: {e.reason}"


@_timed("get_pod_logs")
def get_pod_logs(pod_name: str, namespace: str = "default", tail_lines: int = 50) -> str:
    tail_lines = min(tail_lines, 100)
    try:
        logs = _core.read_namespaced_pod_log(
            name=pod_name, namespace=namespace,
            tail_lines=tail_lines, timestamps=True,
        )
        if not logs.strip():
            return f"No logs for pod '{pod_name}'."
        return f"Last {tail_lines} lines of '{pod_name}':\n{logs}"
    except ApiException as e:
        if e.status == 404:
            return f"Pod '{pod_name}' not found in '{namespace}'."
        return f"K8s API error: {e.reason}"


@_timed("get_events")
def get_events(namespace: str = "default", warning_only: bool = True) -> str:
    try:
        if namespace == "all":
            events = _core.list_event_for_all_namespaces(field_selector="type=Warning" if warning_only else "")
        else:
            events = _core.list_namespaced_event(namespace=namespace, field_selector="type=Warning" if warning_only else "")
        if not events.items:
            return f"No {'warning ' if warning_only else ''}events in '{namespace}'."
        sorted_ev = sorted(events.items, key=lambda e: e.last_timestamp or e.event_time or "", reverse=True)[:20]
        lines = [f"Recent {'warning ' if warning_only else ''}events in '{namespace}':"]
        for ev in sorted_ev:
            lines.append(
                f"  [{ev.type}] {ev.involved_object.kind}/{ev.involved_object.name}: "
                f"{ev.reason} — {ev.message} (x{ev.count or 1})"
            )
        return "\n".join(lines)
    except ApiException as e:
        return f"K8s API error: {e.reason}"


@_timed("get_deployment_status")
def get_deployment_status(namespace: str = "default") -> str:
    try:
        deps = _apps.list_deployment_for_all_namespaces() if namespace == "all" else _apps.list_namespaced_deployment(namespace=namespace)
        if not deps.items:
            return f"No deployments in '{namespace}'."
        lines = [f"Deployments in '{namespace}':"]
        for dep in deps.items:
            desired   = dep.spec.replicas or 0
            ready     = dep.status.ready_replicas or 0
            available = dep.status.available_replicas or 0
            status    = "✓ Healthy" if ready == desired and desired > 0 else "⚠ Degraded"
            lines.append(f"  {dep.metadata.name}: {status} | Desired:{desired} Ready:{ready} Available:{available}")
        return "\n".join(lines)
    except ApiException as e:
        return f"K8s API error: {e.reason}"


@_timed("describe_pod")
def describe_pod(pod_name: str, namespace: str = "default") -> str:
    try:
        pod   = _core.read_namespaced_pod(name=pod_name, namespace=namespace)
        lines = [
            f"Pod: {pod.metadata.name}",
            f"Namespace: {pod.metadata.namespace}",
            f"Phase: {pod.status.phase}",
            "Conditions:",
        ]
        for c in (pod.status.conditions or []):
            lines.append(f"  {c.type}: {c.status}" + (f" — {c.message}" if c.message else ""))
        lines.append("Containers:")
        for cs in (pod.status.container_statuses or []):
            state_key = list(cs.state.to_dict().keys())[0] if cs.state else "unknown"
            lines.append(f"  {cs.name}: ready={cs.ready} restarts={cs.restart_count} state={state_key}")
            if cs.last_state and cs.last_state.terminated:
                lt = cs.last_state.terminated
                lines.append(f"    Last terminated: exit_code={lt.exit_code} reason={lt.reason}")
        for c in pod.spec.containers:
            if c.resources:
                req = c.resources.requests or {}
                lim = c.resources.limits   or {}
                lines.append(
                    f"  {c.name} resources: "
                    f"req=cpu:{req.get('cpu','none')}/mem:{req.get('memory','none')} "
                    f"lim=cpu:{lim.get('cpu','none')}/mem:{lim.get('memory','none')}"
                )
        return "\n".join(lines)
    except ApiException as e:
        if e.status == 404:
            return f"Pod '{pod_name}' not found in '{namespace}'."
        return f"K8s API error: {e.reason}"


K8S_TOOLS = {
    "get_pod_status":        {"fn": get_pod_status,        "description": "List all pods and their status in a namespace.",                          "parameters": {"namespace": {"type":"string","default":"all","description":"Namespace to query. Use 'all' to scan entire cluster (default)"}}},
    "get_node_health":       {"fn": get_node_health,       "description": "Check node health, CPU/memory pressure, and ready status.",               "parameters": {}},
    "get_pod_logs":          {"fn": get_pod_logs,          "description": "Fetch recent logs from a specific pod.",                                  "parameters": {"pod_name":{"type":"string"},"namespace":{"type":"string","default":"default"},"tail_lines":{"type":"integer","default":50}}},
    "get_events":            {"fn": get_events,            "description": "Fetch recent K8s warning events. First step for diagnosing issues.",      "parameters": {"namespace":{"type":"string","default":"default"},"warning_only":{"type":"boolean","default":True}}},
    "get_deployment_status": {"fn": get_deployment_status, "description": "Check deployment replica counts to detect degraded deployments.",         "parameters": {"namespace":{"type":"string","default":"default"}}},
    "describe_pod":          {"fn": describe_pod,          "description": "Get detailed info about a specific pod including container states.",      "parameters": {"pod_name":{"type":"string"},"namespace":{"type":"string","default":"default"}}},
}
