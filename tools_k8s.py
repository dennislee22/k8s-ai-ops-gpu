"""
tools_k8s.py — Kubernetes tool functions and registry
======================================================
All K8s tool functions live here.  To add a new tool:

  1. Write a function that returns a str (plain text for the LLM).
  2. Add an entry to K8S_TOOLS at the bottom of this file.
  3. Add a label to TOOL_LABELS in app.py if you want a nice status badge.

No other file needs to change.

kubectl_exec integration
------------------------
Incorporates the kubectl-ai (GoogleCloudPlatform/kubectl-ai) command execution
model in pure Python:
  - Read/write command classification (mirrors kubectl_filter.go logic)
  - Dangerous-command blocking (edit, port-forward, interactive exec -it)
  - Streaming-command detection (watch, follow-logs, attach)
  - Subprocess execution with KUBECONFIG injection and timeout
  - Output truncation to keep LLM context clean
"""

import os
import re
import shlex
import subprocess
import logging
from pathlib import Path

from kubernetes import client as _k8s, config as _k8s_cfg
from kubernetes.client.rest import ApiException

# ── Logger (uses same name convention as app.py) ──────────────────────────────
_log = logging.getLogger("k8s")


# ─────────────────────────────────────────────────────────────────────────────
# CLIENT INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────

def _load_k8s():
    kc = os.getenv("KUBECONFIG_PATH", "")
    try:
        if kc and Path(os.path.expanduser(kc)).exists():
            _k8s_cfg.load_kube_config(config_file=os.path.expanduser(kc))
            _log.info(f"Loaded kubeconfig: {kc}")
        else:
            _k8s_cfg.load_incluster_config()
            _log.info("Loaded in-cluster config")
    except Exception as e:
        _log.error(f"K8s config failed: {e}")
        raise RuntimeError(f"K8s config: {e}")


_load_k8s()

_core   = _k8s.CoreV1Api()
_apps   = _k8s.AppsV1Api()
_batch  = _k8s.BatchV1Api()
_rbac   = _k8s.RbacAuthorizationV1Api()
_net    = _k8s.NetworkingV1Api()
_autoscaling = _k8s.AutoscalingV2Api()


# ─────────────────────────────────────────────────────────────────────────────
# TOOL FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

# ── Pods ──────────────────────────────────────────────────────────────────────

def get_pod_status(namespace: str = "all", show_all: bool = False) -> str:
    """
    List pods in a namespace.

    By default (show_all=False) only non-healthy pods are returned — pods that
    are Running, fully ready, and have 0 restarts are omitted to keep output
    concise for health-check queries.

    Set show_all=True (or ask "how many pods", "list all pods", "list pods in
    <namespace>") to return EVERY pod including healthy ones.  Always use
    show_all=True when the user asks for a count or a complete pod listing.
    """
    try:
        pods = (_core.list_pod_for_all_namespaces() if namespace == "all"
                else _core.list_namespaced_pod(namespace=namespace))
        if not pods.items:
            return f"No pods found in namespace '{namespace}'."

        lines   = [f"Pods in '{namespace}' (total: {len(pods.items)}):"]
        healthy = []
        unhealthy = []

        for pod in pods.items:
            phase    = pod.status.phase or "Unknown"
            restarts = sum(cs.restart_count for cs in (pod.status.container_statuses or []))
            ready    = sum(1 for cs in (pod.status.container_statuses or []) if cs.ready)
            total    = len(pod.spec.containers)
            bad      = [f"{c.type}={c.status}"
                        for c in (pod.status.conditions or []) if c.status != "True"]
            row = (
                f"  {pod.metadata.namespace}/{pod.metadata.name}: {phase} "
                f"| Ready {ready}/{total} | Restarts:{restarts}"
                + (f" [{', '.join(bad)}]" if bad else "")
            )
            is_ok = (phase in ("Running", "Succeeded", "Completed")
                     and ready == total and restarts == 0)
            if is_ok:
                healthy.append(row)
            else:
                unhealthy.append(row)

        if show_all:
            lines += unhealthy + healthy
        else:
            lines += unhealthy
            if healthy:
                lines.append(f"  ({len(healthy)} healthy pod(s) omitted — use show_all=true to list them)")

        if len(lines) == 1:
            # Only the header — everything was healthy
            return (f"All {len(pods.items)} pod(s) healthy in namespace '{namespace}'.\n"
                    + "\n".join(healthy))
        return "\n".join(lines)
    except ApiException as e:
        return f"K8s API error: {e.reason}"


def get_pod_logs(pod_name: str, namespace: str = "default",
                 tail_lines: int = 50) -> str:
    tail_lines = min(tail_lines, 100)
    try:
        logs = _core.read_namespaced_pod_log(
            name=pod_name, namespace=namespace,
            tail_lines=tail_lines, timestamps=True)
        return (f"Last {tail_lines} lines of '{pod_name}':\n{logs}"
                if logs.strip() else f"No logs for '{pod_name}'.")
    except ApiException as e:
        return (f"Pod '{pod_name}' not found."
                if e.status == 404 else f"K8s error: {e.reason}")


def describe_pod(pod_name: str, namespace: str = "default") -> str:
    try:
        pod   = _core.read_namespaced_pod(name=pod_name, namespace=namespace)
        lines = [
            f"Pod:       {pod.metadata.name}",
            f"Namespace: {pod.metadata.namespace}",
            f"Phase:     {pod.status.phase}",
            "Conditions:",
        ]
        for c in (pod.status.conditions or []):
            lines.append(f"  {c.type}:{c.status}"
                         + (f" — {c.message}" if c.message else ""))
        lines.append("Containers:")
        for cs in (pod.status.container_statuses or []):
            sk = list(cs.state.to_dict().keys())[0] if cs.state else "unknown"
            lines.append(f"  {cs.name}: ready={cs.ready} "
                         f"restarts={cs.restart_count} state={sk}")
            if cs.last_state and cs.last_state.terminated:
                lt = cs.last_state.terminated
                lines.append(f"    Last terminated: exit={lt.exit_code} "
                              f"reason={lt.reason}")
        for c in pod.spec.containers:
            if c.resources:
                req = c.resources.requests or {}
                lim = c.resources.limits   or {}
                lines.append(
                    f"  {c.name} resources: "
                    f"req=cpu:{req.get('cpu','none')}/mem:{req.get('memory','none')} "
                    f"lim=cpu:{lim.get('cpu','none')}/mem:{lim.get('memory','none')}")
        return "\n".join(lines)
    except ApiException as e:
        return (f"Pod '{pod_name}' not found."
                if e.status == 404 else f"K8s error: {e.reason}")


# ── Nodes ─────────────────────────────────────────────────────────────────────

def get_node_health() -> str:
    try:
        nodes = _core.list_node()
        if not nodes.items:
            return "No nodes found."
        lines = ["Node health:"]
        for node in nodes.items:
            roles    = [k.replace("node-role.kubernetes.io/", "")
                        for k in (node.metadata.labels or {})
                        if k.startswith("node-role.kubernetes.io/")] or ["worker"]
            conds    = {c.type: c.status for c in (node.status.conditions or [])}
            pressure = [t for t in ["MemoryPressure", "DiskPressure", "PIDPressure"]
                        if conds.get(t) == "True"]
            alloc    = node.status.allocatable or {}
            lines.append(
                f"  {node.metadata.name} [{','.join(roles)}]: "
                f"Ready={conds.get('Ready','?')}"
                + (f" ⚠ {','.join(pressure)}" if pressure else "")
                + f" | CPU:{alloc.get('cpu','n/a')} Mem:{alloc.get('memory','n/a')}")
        return "\n".join(lines)
    except ApiException as e:
        return f"K8s API error: {e.reason}"


# ── Events ────────────────────────────────────────────────────────────────────

# Event messages to suppress at tool level — these are environment noise that
# the LLM is instructed to ignore anyway, but filtering here prevents the raw
# text from leaking into LLM context and triggering confusing responses.
_EVENT_NOISE_PATTERNS = [
    "cgroup",       # cgroupv1/v2 maintenance mode warnings — environment constraint, not actionable
    "cgroupv",
    "cgroup v1",
    "cgroup v2",
]

def _is_noisy_event(message: str) -> bool:
    """Return True if the event message is known background noise."""
    msg_lower = (message or "").lower()
    return any(pat in msg_lower for pat in _EVENT_NOISE_PATTERNS)


def get_events(namespace: str = "all", warning_only: bool = True) -> str:
    try:
        fs = "type=Warning" if warning_only else ""
        ev = (_core.list_event_for_all_namespaces(field_selector=fs, limit=500)
              if namespace == "all"
              else _core.list_namespaced_event(namespace=namespace,
                                               field_selector=fs, limit=500))
        if not ev.items:
            return f"No {'warning ' if warning_only else ''}events in '{namespace}'."

        # Sort newest-first, cap at 20, suppress known noise
        sev = sorted(ev.items,
                     key=lambda e: e.last_timestamp or e.event_time or "",
                     reverse=True)

        lines      = [f"Recent events in '{namespace}':"]
        shown      = 0
        suppressed = 0
        for e in sev:
            if shown >= 20:
                break
            if _is_noisy_event(e.message):
                suppressed += 1
                continue
            lines.append(
                f"  [{e.type}] {e.involved_object.kind}/{e.involved_object.name}: "
                f"{e.reason} — {e.message} (x{e.count or 1})")
            shown += 1

        if suppressed:
            lines.append(f"  ({suppressed} environment-noise event(s) suppressed)")
        if shown == 0:
            return f"No actionable events in '{namespace}' (all were background noise)."
        return "\n".join(lines)
    except ApiException as e:
        return f"K8s API error: {e.reason}"


# ── Workloads ─────────────────────────────────────────────────────────────────

def get_deployment_status(namespace: str = "all") -> str:
    try:
        deps = (_apps.list_deployment_for_all_namespaces()
                if namespace == "all"
                else _apps.list_namespaced_deployment(namespace=namespace))
        if not deps.items:
            return f"No deployments in '{namespace}'."
        lines = [f"Deployments in '{namespace}':"]
        for dep in deps.items:
            desired = dep.spec.replicas or 0
            ready   = dep.status.ready_replicas or 0
            avail   = dep.status.available_replicas or 0
            status  = "✓ Healthy" if ready == desired and desired > 0 else "⚠ Degraded"
            lines.append(
                f"  {dep.metadata.namespace}/{dep.metadata.name}: {status} "
                f"| Desired:{desired} Ready:{ready} Available:{avail}")
        return "\n".join(lines)
    except ApiException as e:
        return f"K8s API error: {e.reason}"


def get_daemonset_status(namespace: str = "all") -> str:
    try:
        ds = (_apps.list_daemon_set_for_all_namespaces()
              if namespace == "all"
              else _apps.list_namespaced_daemon_set(namespace=namespace))
        if not ds.items:
            return f"No DaemonSets in '{namespace}'."
        lines = [f"DaemonSets in '{namespace}':"]
        for d in ds.items:
            desired   = d.status.desired_number_scheduled or 0
            ready     = d.status.number_ready or 0
            available = d.status.number_available or 0
            status    = "✓ Healthy" if ready == desired and desired > 0 else "⚠ Degraded"
            lines.append(
                f"  {d.metadata.namespace}/{d.metadata.name}: {status} "
                f"| Desired:{desired} Ready:{ready} Available:{available}")
        return "\n".join(lines)
    except ApiException as e:
        return f"K8s API error: {e.reason}"


def get_statefulset_status(namespace: str = "all") -> str:
    try:
        sts = (_apps.list_stateful_set_for_all_namespaces()
               if namespace == "all"
               else _apps.list_namespaced_stateful_set(namespace=namespace))
        if not sts.items:
            return f"No StatefulSets in '{namespace}'."
        lines = [f"StatefulSets in '{namespace}':"]
        for s in sts.items:
            desired = s.spec.replicas or 0
            ready   = s.status.ready_replicas or 0
            status  = "✓ Healthy" if ready == desired and desired > 0 else "⚠ Degraded"
            lines.append(
                f"  {s.metadata.namespace}/{s.metadata.name}: {status} "
                f"| Desired:{desired} Ready:{ready}")
        return "\n".join(lines)
    except ApiException as e:
        return f"K8s API error: {e.reason}"


def get_job_status(namespace: str = "all") -> str:
    try:
        jobs = (_batch.list_job_for_all_namespaces()
                if namespace == "all"
                else _batch.list_namespaced_job(namespace=namespace))
        if not jobs.items:
            return f"No Jobs in '{namespace}'."
        lines = [f"Jobs in '{namespace}':"]
        for j in jobs.items:
            active    = j.status.active    or 0
            succeeded = j.status.succeeded or 0
            failed    = j.status.failed    or 0
            status    = ("✓ Complete" if succeeded > 0 and active == 0
                         else "⚠ Failed" if failed > 0
                         else "⏳ Running")
            lines.append(
                f"  {j.metadata.namespace}/{j.metadata.name}: {status} "
                f"| Active:{active} Succeeded:{succeeded} Failed:{failed}")
        return "\n".join(lines)
    except ApiException as e:
        return f"K8s API error: {e.reason}"


def get_hpa_status(namespace: str = "all") -> str:
    """Check HorizontalPodAutoscaler targets and current replica counts."""
    try:
        hpas = (_autoscaling.list_horizontal_pod_autoscaler_for_all_namespaces()
                if namespace == "all"
                else _autoscaling.list_namespaced_horizontal_pod_autoscaler(
                    namespace=namespace))
        if not hpas.items:
            return f"No HPAs in '{namespace}'."
        lines = [f"HPAs in '{namespace}':"]
        for h in hpas.items:
            cur = h.status.current_replicas or 0
            des = h.status.desired_replicas or 0
            mn  = h.spec.min_replicas or 1
            mx  = h.spec.max_replicas or "?"
            at_max = cur >= h.spec.max_replicas if h.spec.max_replicas else False
            flag = " ⚠ AT MAX" if at_max else ""
            lines.append(
                f"  {h.metadata.namespace}/{h.metadata.name}: "
                f"current={cur} desired={des} min={mn} max={mx}{flag}")
        return "\n".join(lines)
    except ApiException as e:
        return f"K8s API error: {e.reason}"


# ── Storage ───────────────────────────────────────────────────────────────────

def get_pvc_status(namespace: str = "all") -> str:
    """Check PersistentVolumeClaims — highlights Pending or Lost claims."""
    try:
        pvcs = (_core.list_persistent_volume_claim_for_all_namespaces()
                if namespace == "all"
                else _core.list_namespaced_persistent_volume_claim(
                    namespace=namespace))
        if not pvcs.items:
            return f"No PVCs in '{namespace}'."
        lines = [f"PVCs in '{namespace}':"]
        skipped = 0
        for pvc in pvcs.items:
            phase = pvc.status.phase or "Unknown"
            if phase == "Bound":
                skipped += 1; continue
            sc  = pvc.spec.storage_class_name or "default"
            cap = (pvc.status.capacity or {}).get("storage", "?")
            lines.append(
                f"  {pvc.metadata.namespace}/{pvc.metadata.name}: "
                f"{phase} | class:{sc} capacity:{cap}")
        if skipped:
            lines.append(f"  ({skipped} Bound PVCs omitted)")
        return "\n".join(lines) if len(lines) > 1 else f"All PVCs Bound in '{namespace}'."
    except ApiException as e:
        return f"K8s API error: {e.reason}"


def get_persistent_volumes() -> str:
    """List all PersistentVolumes with phase, capacity, and reclaim policy."""
    try:
        pvs = _core.list_persistent_volume()
        if not pvs.items:
            return "No PersistentVolumes found."
        lines = ["PersistentVolumes:"]
        for pv in pvs.items:
            phase    = pv.status.phase or "Unknown"
            cap      = (pv.spec.capacity or {}).get("storage", "?")
            policy   = pv.spec.persistent_volume_reclaim_policy or "?"
            sc       = pv.spec.storage_class_name or "none"
            claim    = (f"{pv.spec.claim_ref.namespace}/{pv.spec.claim_ref.name}"
                        if pv.spec.claim_ref else "unbound")
            lines.append(
                f"  {pv.metadata.name}: {phase} | {cap} | "
                f"class:{sc} policy:{policy} claim:{claim}")
        return "\n".join(lines)
    except ApiException as e:
        return f"K8s API error: {e.reason}"


# ── Networking ────────────────────────────────────────────────────────────────

def get_service_status(namespace: str = "all") -> str:
    """List Services — highlights those with no endpoints (potential misconfigs)."""
    try:
        svcs = (_core.list_service_for_all_namespaces()
                if namespace == "all"
                else _core.list_namespaced_service(namespace=namespace))
        if not svcs.items:
            return f"No services in '{namespace}'."
        lines = [f"Services in '{namespace}':"]
        for svc in svcs.items:
            stype    = svc.spec.type or "ClusterIP"
            ports    = ", ".join(
                f"{p.port}/{p.protocol}" for p in (svc.spec.ports or []))
            selector = svc.spec.selector or {}
            flag     = "" if selector else " ⚠ no selector"
            lines.append(
                f"  {svc.metadata.namespace}/{svc.metadata.name}: "
                f"{stype} ports:[{ports}]{flag}")
        return "\n".join(lines)
    except ApiException as e:
        return f"K8s API error: {e.reason}"


def get_ingress_status(namespace: str = "all") -> str:
    try:
        ings = (_net.list_ingress_for_all_namespaces()
                if namespace == "all"
                else _net.list_namespaced_ingress(namespace=namespace))
        if not ings.items:
            return f"No Ingresses in '{namespace}'."
        lines = [f"Ingresses in '{namespace}':"]
        for ing in ings.items:
            cls   = ing.spec.ingress_class_name or "default"
            hosts = [rule.host or "*" for rule in (ing.spec.rules or [])]
            lb    = [
                addr.ip or addr.hostname
                for status in (ing.status.load_balancer.ingress or [])
                for addr in [status]
                if status
            ] if ing.status.load_balancer else []
            lines.append(
                f"  {ing.metadata.namespace}/{ing.metadata.name}: "
                f"class:{cls} hosts:{hosts} lb:{lb or 'pending'}")
        return "\n".join(lines)
    except ApiException as e:
        return f"K8s API error: {e.reason}"


# ── Config & Resources ────────────────────────────────────────────────────────

def get_configmap_list(namespace: str = "default") -> str:
    """List ConfigMaps in a namespace (excludes kube-system defaults)."""
    try:
        cms = _core.list_namespaced_config_map(namespace=namespace)
        skip = {"kube-root-ca.crt"}
        items = [cm for cm in cms.items if cm.metadata.name not in skip]
        if not items:
            return f"No ConfigMaps in '{namespace}'."
        lines = [f"ConfigMaps in '{namespace}':"]
        for cm in items:
            keys = list((cm.data or {}).keys())
            lines.append(f"  {cm.metadata.name}: keys={keys}")
        return "\n".join(lines)
    except ApiException as e:
        return f"K8s API error: {e.reason}"


def get_resource_quotas(namespace: str = "all") -> str:
    """Check ResourceQuotas — highlights namespaces near their limits."""
    try:
        quotas = (_core.list_resource_quota_for_all_namespaces()
                  if namespace == "all"
                  else _core.list_namespaced_resource_quota(namespace=namespace))
        if not quotas.items:
            return f"No ResourceQuotas in '{namespace}'."
        lines = [f"ResourceQuotas in '{namespace}':"]
        for q in quotas.items:
            hard = q.status.hard or {}
            used = q.status.used or {}
            lines.append(f"  {q.metadata.namespace}/{q.metadata.name}:")
            for resource, limit in hard.items():
                current = used.get(resource, "0")
                lines.append(f"    {resource}: {current} / {limit}")
        return "\n".join(lines)
    except ApiException as e:
        return f"K8s API error: {e.reason}"


def get_limit_ranges(namespace: str = "all") -> str:
    try:
        lrs = (_core.list_limit_range_for_all_namespaces()
               if namespace == "all"
               else _core.list_namespaced_limit_range(namespace=namespace))
        if not lrs.items:
            return f"No LimitRanges in '{namespace}'."
        lines = [f"LimitRanges in '{namespace}':"]
        for lr in lrs.items:
            lines.append(f"  {lr.metadata.namespace}/{lr.metadata.name}:")
            for item in (lr.spec.limits or []):
                lines.append(
                    f"    type:{item.type} "
                    f"max:{item.max} min:{item.min} default:{item.default}")
        return "\n".join(lines)
    except ApiException as e:
        return f"K8s API error: {e.reason}"


# ── RBAC ──────────────────────────────────────────────────────────────────────

def get_service_accounts(namespace: str = "default") -> str:
    try:
        sas = _core.list_namespaced_service_account(namespace=namespace)
        if not sas.items:
            return f"No ServiceAccounts in '{namespace}'."
        lines = [f"ServiceAccounts in '{namespace}':"]
        for sa in sas.items:
            secrets = len(sa.secrets or [])
            lines.append(f"  {sa.metadata.name}: secrets={secrets}")
        return "\n".join(lines)
    except ApiException as e:
        return f"K8s API error: {e.reason}"


def get_cluster_role_bindings() -> str:
    """List ClusterRoleBindings — useful for auditing broad permissions."""
    try:
        crbs = _rbac.list_cluster_role_binding()
        if not crbs.items:
            return "No ClusterRoleBindings found."
        lines = ["ClusterRoleBindings:"]
        for crb in crbs.items:
            role     = crb.role_ref.name
            subjects = [f"{s.kind}/{s.name}" for s in (crb.subjects or [])]
            lines.append(f"  {crb.metadata.name}: role={role} → {subjects}")
        return "\n".join(lines)
    except ApiException as e:
        return f"K8s API error: {e.reason}"


# ── Namespaces ────────────────────────────────────────────────────────────────

def get_namespace_status() -> str:
    """
    List ALL namespaces and their phase/status.

    Uses kubectl directly to bypass the Python SDK's silent pagination cap,
    which can under-report namespace counts on large clusters.
    """
    result = kubectl_exec("kubectl get namespaces -o wide")
    if result.startswith("[ERROR]"):
        # Fallback to SDK if kubectl is unavailable
        try:
            nss   = _core.list_namespace(limit=1000)
            lines = [f"Namespaces (total: {len(nss.items)}):"]
            for ns in nss.items:
                phase = ns.status.phase or "Unknown"
                lines.append(f"  {ns.metadata.name}: {phase}")
            return "\n".join(lines)
        except ApiException as e:
            return f"K8s API error: {e.reason}"
    # Prepend a total-count header so the LLM always sees the real number
    rows = [r for r in result.strip().splitlines() if r and not r.startswith("NAME")]
    header = f"Namespaces (total: {len(rows)}):\n"
    return header + result


# ─────────────────────────────────────────────────────────────────────────────
# KUBECTL EXEC — kubectl-ai command execution layer (pure Python port)
#
# Ported from GoogleCloudPlatform/kubectl-ai:
#   pkg/tools/kubectl_tool.go  — command execution & validation
#   pkg/tools/kubectl_filter.go — read/write classification
#   pkg/tools/bash_tool.go     — streaming detection
#
# Design decisions:
#   • READ-ONLY by default: write ops are blocked unless KUBECTL_ALLOW_WRITES=true
#   • Dangerous interactive/streaming commands are always blocked
#   • KUBECONFIG is injected from KUBECONFIG_PATH env (same source as SDK tools)
#   • Output is truncated to MAX_OUTPUT_CHARS to protect LLM context window
#   • Timeout defaults to 30 s; override with KUBECTL_TIMEOUT env var
# ─────────────────────────────────────────────────────────────────────────────

# ── Constants (mirrors kubectl_filter.go) ─────────────────────────────────────

_KUBECTL_READ_ONLY_OPS = {
    "get", "describe", "explain", "top", "logs", "api-resources",
    "api-versions", "version", "config", "cluster-info", "wait", "auth",
    "diff", "kustomize", "help", "options", "proxy", "completion",
    "convert", "events", "can-i", "whoami",
}

_KUBECTL_WRITE_OPS = {
    "create", "apply", "edit", "delete", "patch", "replace", "scale",
    "autoscale", "expose", "run", "exec", "set", "label", "annotate",
    "taint", "drain", "cordon", "uncordon", "debug", "attach", "cp",
    "reconcile", "approve", "deny", "certificate",
}

# rollout sub-commands that are read-only
_ROLLOUT_READ_ONLY = {"history", "status"}

# Commands that are always blocked regardless of read/write mode
_BLOCKED_PATTERNS = [
    ("kubectl edit",         "interactive mode not supported; use 'kubectl get -o yaml' + 'kubectl apply' instead"),
    ("kubectl port-forward", "port-forward is not supported in unattended mode"),
    ("kubectl exec -it",     "interactive exec not supported; use 'kubectl exec -- <cmd>' without -it"),
    ("kubectl exec -ti",     "interactive exec not supported; use 'kubectl exec -- <cmd>' without -ti"),
    ("kubectl attach",       "interactive attach not supported"),
]

# Streaming commands that cannot complete in a fixed timeout — always blocked
_STREAMING_PATTERNS = [
    (lambda cmd: " get " in cmd and " -w" in cmd,    "watch mode (-w) not supported"),
    (lambda cmd: " logs " in cmd and " -f" in cmd,   "follow-log mode (-f) not supported; use tail_lines instead"),
    (lambda cmd: " attach " in cmd,                  "attach is not supported"),
]

_KUBECTL_TIMEOUT   = int(os.getenv("KUBECTL_TIMEOUT",  "30"))
_KUBECTL_MAX_OUT   = int(os.getenv("KUBECTL_MAX_CHARS", "8000"))
_ALLOW_WRITES      = os.getenv("KUBECTL_ALLOW_WRITES", "false").lower() in ("1", "true", "yes")


def _kubectl_modifies_resource(command: str) -> str:
    """
    Classify a kubectl command as 'yes' / 'no' / 'unknown' for write impact.
    Mirrors kubectlModifiesResource() in kubectl_filter.go.
    """
    tokens = shlex.split(command) if command.strip() else []
    # Strip leading env-var assignments / paths
    kubectl_idx = next((i for i, t in enumerate(tokens)
                        if os.path.basename(t) == "kubectl"), None)
    if kubectl_idx is None:
        return "unknown"
    sub_tokens = tokens[kubectl_idx + 1:]
    # Skip flags that precede the sub-command
    ops = [t for t in sub_tokens if not t.startswith("-")]
    if not ops:
        return "unknown"
    verb = ops[0]
    if verb in _KUBECTL_READ_ONLY_OPS:
        # rollout has mixed sub-commands: the sub-verb follows 'rollout'
        # e.g. "kubectl rollout history deployment/app" → ops = ["rollout", "history", "deployment/app"]
        if verb == "rollout":
            # find sub-verb: first token after 'rollout' that doesn't contain '/'
            sub_ops = [t for t in ops[1:] if "/" not in t]
            if sub_ops:
                return "no" if sub_ops[0] in _ROLLOUT_READ_ONLY else "yes"
            return "unknown"
        return "no"
    if verb in _KUBECTL_WRITE_OPS:
        return "yes"
    return "unknown"


def _validate_kubectl_command(command: str) -> str | None:
    """
    Return an error string if the command is blocked, else None.
    Mirrors validateKubectlCommand() + streaming detection.
    """
    cmd_lower = command.lower()
    for pattern, reason in _BLOCKED_PATTERNS:
        if pattern in command:
            return reason
    for check_fn, reason in _STREAMING_PATTERNS:
        if check_fn(command):
            return reason
    return None


def kubectl_exec(command: str) -> str:
    """
    Execute a kubectl command against the cluster and return its output.

    Implements the kubectl-ai execution model:
      • Dangerous / interactive commands are rejected before execution.
      • Write operations are blocked unless KUBECTL_ALLOW_WRITES=true.
      • KUBECONFIG is injected from KUBECONFIG_PATH env var.
      • Execution is time-bounded (KUBECTL_TIMEOUT seconds, default 30).
      • Stdout + stderr are merged and truncated to KUBECTL_MAX_CHARS chars.

    Parameters
    ----------
    command : str
        The full kubectl command string, e.g. ``kubectl get pods -n default``.
        Must begin with ``kubectl``.

    Returns
    -------
    str
        Command output, or an error message prefixed with ``[ERROR]``.
    """
    command = command.strip()

    # ── 1. Must start with kubectl ────────────────────────────────────────────
    if not re.match(r"^(sudo\s+)?kubectl(\s|$)", command):
        return "[ERROR] kubectl_exec only runs kubectl commands. Command must start with 'kubectl'."

    # ── 2. Validate: blocked / streaming ─────────────────────────────────────
    err = _validate_kubectl_command(command)
    if err:
        return f"[ERROR] Command not allowed: {err}"

    # ── 3. Write-protection ───────────────────────────────────────────────────
    if not _ALLOW_WRITES and _kubectl_modifies_resource(command) == "yes":
        return (
            "[ERROR] Write operations are disabled (KUBECTL_ALLOW_WRITES is not set). "
            "This tool is read-only by default. If you need to make changes, ask the "
            "operator to set KUBECTL_ALLOW_WRITES=true."
        )

    # ── 4. Build environment with KUBECONFIG ──────────────────────────────────
    env = os.environ.copy()
    kc_path = os.getenv("KUBECONFIG_PATH", "")
    if kc_path:
        expanded = os.path.expanduser(kc_path)
        if Path(expanded).exists():
            env["KUBECONFIG"] = expanded

    # ── 5. Execute ────────────────────────────────────────────────────────────
    _log.info(f"[kubectl_exec] Running: {command!r}")
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            env=env, timeout=_KUBECTL_TIMEOUT,
        )
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        combined = (stdout + ("\n" + stderr if stderr.strip() else "")).strip()

        if result.returncode != 0 and not combined:
            combined = f"[exit {result.returncode}] (no output)"
        elif result.returncode != 0:
            combined = f"[exit {result.returncode}]\n{combined}"

        # Truncate to protect LLM context window
        if len(combined) > _KUBECTL_MAX_OUT:
            combined = combined[:_KUBECTL_MAX_OUT] + f"\n...[output truncated at {_KUBECTL_MAX_OUT} chars]"

        return combined or "(empty output)"

    except subprocess.TimeoutExpired:
        return f"[ERROR] Command timed out after {_KUBECTL_TIMEOUT}s: {command}"
    except FileNotFoundError:
        return "[ERROR] 'kubectl' binary not found. Ensure kubectl is installed and on PATH."
    except Exception as exc:
        _log.exception(f"[kubectl_exec] Unexpected error: {exc}")
        return f"[ERROR] Unexpected error: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# TOOL REGISTRY
# Add entries here to expose a function to the LLM agent.
# ─────────────────────────────────────────────────────────────────────────────

K8S_TOOLS: dict = {

    # ── Pods ──────────────────────────────────────────────────────────────────
    "get_pod_status": {
        "fn":          get_pod_status,
        "description": (
            "List pods in a namespace. "
            "By default only UNHEALTHY pods are returned (non-Running, not ready, or high restarts). "
            "Set show_all=true to list ALL pods including healthy ones — ALWAYS use show_all=true "
            "when the user asks 'how many pods', 'list pods', 'what pods are running', or "
            "any question that requires a complete pod count or inventory."
        ),
        "parameters":  {
            "namespace": {"type": "string",  "default": "all"},
            "show_all":  {"type": "boolean", "default": False,
                          "description": "Set true to include healthy/running pods in the output"},
        },
    },
    "get_pod_logs": {
        "fn":          get_pod_logs,
        "description": "Fetch recent logs from a specific pod.",
        "parameters":  {
            "pod_name":   {"type": "string"},
            "namespace":  {"type": "string",  "default": "default"},
            "tail_lines": {"type": "integer", "default": 50},
        },
    },
    "describe_pod": {
        "fn":          describe_pod,
        "description": "Get detailed info about a specific pod including container states and resource limits.",
        "parameters":  {
            "pod_name":  {"type": "string"},
            "namespace": {"type": "string", "default": "default"},
        },
    },

    # ── Nodes ─────────────────────────────────────────────────────────────────
    "get_node_health": {
        "fn":          get_node_health,
        "description": "Check node health, CPU/memory/disk pressure, and allocatable resources.",
        "parameters":  {},
    },

    # ── Events ────────────────────────────────────────────────────────────────
    "get_events": {
        "fn":          get_events,
        "description": "Fetch recent K8s events. Always the first step for diagnosing issues.",
        "parameters":  {
            "namespace":    {"type": "string",  "default": "all"},
            "warning_only": {"type": "boolean", "default": True},
        },
    },

    # ── Workloads ─────────────────────────────────────────────────────────────
    "get_deployment_status": {
        "fn":          get_deployment_status,
        "description": "Check Deployment replica counts and health across namespaces.",
        "parameters":  {"namespace": {"type": "string", "default": "all"}},
    },
    "get_daemonset_status": {
        "fn":          get_daemonset_status,
        "description": "Check DaemonSet scheduling health — useful for node-level agents (e.g. Longhorn, CNI).",
        "parameters":  {"namespace": {"type": "string", "default": "all"}},
    },
    "get_statefulset_status": {
        "fn":          get_statefulset_status,
        "description": "Check StatefulSet replica counts — useful for databases and Longhorn components.",
        "parameters":  {"namespace": {"type": "string", "default": "all"}},
    },
    "get_job_status": {
        "fn":          get_job_status,
        "description": "Check batch Job and CronJob run status — highlights failed jobs.",
        "parameters":  {"namespace": {"type": "string", "default": "all"}},
    },
    "get_hpa_status": {
        "fn":          get_hpa_status,
        "description": "Check HorizontalPodAutoscaler targets and whether any are pinned at max replicas.",
        "parameters":  {"namespace": {"type": "string", "default": "all"}},
    },

    # ── Storage ───────────────────────────────────────────────────────────────
    "get_pvc_status": {
        "fn":          get_pvc_status,
        "description": "Check PersistentVolumeClaims — highlights Pending or Lost claims. Use namespace='longhorn-system' for Longhorn storage issues.",
        "parameters":  {"namespace": {"type": "string", "default": "all"}},
    },
    "get_persistent_volumes": {
        "fn":          get_persistent_volumes,
        "description": "List all PersistentVolumes with phase, capacity, and reclaim policy.",
        "parameters":  {},
    },

    # ── Networking ────────────────────────────────────────────────────────────
    "get_service_status": {
        "fn":          get_service_status,
        "description": "List Services and highlight those with no pod selector (potential misconfigs).",
        "parameters":  {"namespace": {"type": "string", "default": "all"}},
    },
    "get_ingress_status": {
        "fn":          get_ingress_status,
        "description": "List Ingress rules, hostnames, and load balancer addresses.",
        "parameters":  {"namespace": {"type": "string", "default": "all"}},
    },

    # ── Config & Resources ────────────────────────────────────────────────────
    "get_configmap_list": {
        "fn":          get_configmap_list,
        "description": "List ConfigMaps in a namespace — useful for checking configuration drift.",
        "parameters":  {"namespace": {"type": "string", "default": "default"}},
    },
    "get_resource_quotas": {
        "fn":          get_resource_quotas,
        "description": "Check ResourceQuotas and current usage — useful when pods fail to schedule.",
        "parameters":  {"namespace": {"type": "string", "default": "all"}},
    },
    "get_limit_ranges": {
        "fn":          get_limit_ranges,
        "description": "List LimitRanges that enforce default CPU/memory constraints per namespace.",
        "parameters":  {"namespace": {"type": "string", "default": "all"}},
    },

    # ── RBAC ──────────────────────────────────────────────────────────────────
    "get_service_accounts": {
        "fn":          get_service_accounts,
        "description": "List ServiceAccounts in a namespace.",
        "parameters":  {"namespace": {"type": "string", "default": "default"}},
    },
    "get_cluster_role_bindings": {
        "fn":          get_cluster_role_bindings,
        "description": "List ClusterRoleBindings — useful for auditing broad RBAC permissions.",
        "parameters":  {},
    },

    # ── Namespaces ────────────────────────────────────────────────────────────
    "get_namespace_status": {
        "fn":          get_namespace_status,
        "description": "List all namespaces and their phase.",
        "parameters":  {},
    },
}

# ── kubectl_exec patch: appended after initial registry definition ─────────────
K8S_TOOLS["kubectl_exec"] = {
    "fn":          kubectl_exec,
    "description": (
        "Execute any kubectl command against the cluster. "
        "Use for operations not covered by other tools: custom resources (CRDs), "
        "Longhorn volumes/replicas/engines, rollout history/status, top nodes/pods, "
        "auth can-i, api-resources, diff, and ad-hoc read-only diagnostics. "
        "WRITE commands (apply, delete, patch, scale, etc.) are blocked unless "
        "KUBECTL_ALLOW_WRITES=true is set by the operator. "
        "Interactive commands (exec -it, logs -f, port-forward) are always blocked. "
        "Always prefix the command with 'kubectl'."
    ),
    "parameters": {
        "command": {
            "type": "string",
            "description": (
                "Full kubectl command starting with 'kubectl'. "
                "Examples: 'kubectl get pods -A', "
                "'kubectl describe node worker-1', "
                "'kubectl get volumes.longhorn.io -n longhorn-system', "
                "'kubectl rollout history deployment/myapp -n prod', "
                "'kubectl top nodes', "
                "'kubectl auth can-i list pods --as system:serviceaccount:default:mysa'"
            ),
        },
    },
}
