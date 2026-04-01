#!/usr/bin/env python3
"""Extract verdict, reason, or params from OpenClaw agent sandbox output.

Usage:
    echo "$AGENT_OUTPUT" | python3 extract_agent_output.py verdict
    echo "$AGENT_OUTPUT" | python3 extract_agent_output.py reason
    echo "$AGENT_OUTPUT" | python3 extract_agent_output.py params [backend]
"""
import sys
import json
import re


def unwrap_openclaw_payload(raw: str) -> str:
    """Extract the LLM text from OpenClaw JSON wrapper, or clean raw text."""
    # Strip ANSI escape sequences
    raw = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", raw)

    # Find OpenClaw JSON wrapper using balanced-brace matching
    for i, c in enumerate(raw):
        if c != "{":
            continue
        depth = 0
        for j in range(i, len(raw)):
            if raw[j] == "{":
                depth += 1
            elif raw[j] == "}":
                depth -= 1
            if depth == 0:
                candidate = raw[i : j + 1]
                try:
                    obj = json.loads(candidate)
                    if "payloads" in obj:
                        text = obj.get("payloads", [{}])[0].get("text", "")
                        if text:
                            return text
                except Exception:
                    pass
                break

    # Fallback: strip openshell noise lines
    skip = [
        "Requesting sandbox", "Pulling image", "Starting sandbox",
        "Sandbox allocated", "Sandbox ready", "Image pulled",
        "Uploading files", "Files uploaded", "Deleted sandbox",
        "UNDICI-EHPA", "Created sandbox",
    ]
    lines = []
    for line in raw.splitlines():
        s = line.strip()
        if s and not any(x in s for x in skip):
            lines.append(line)
    return "\n".join(lines)


def find_verdict_json(text: str):
    """Find JSON objects containing a verdict key."""
    cleaned = re.sub(r"`{3}[a-z]*\n?", "", text)
    for m in reversed(list(re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", cleaned))):
        try:
            obj = json.loads(m.group())
            if "verdict" in obj:
                return obj
        except Exception:
            continue
    return None


def extract_verdict(text: str) -> str:
    obj = find_verdict_json(text)
    if obj:
        v = obj.get("verdict", "").upper()
        if v == "ACCEPT":
            return "ACCEPT"
        if v in ("ITERATE", "FAIL", "REJECT", "RETRY"):
            return "ITERATE"

    # Regex fallback on raw text
    if re.search(r"\bACCEPT\b", text):
        return "ACCEPT"
    if re.search(r"\bITERATE\b|\bFAIL\b|\bREJECT\b", text):
        return "ITERATE"
    return "UNKNOWN"


def extract_reason(text: str) -> str:
    obj = find_verdict_json(text)
    if obj:
        for key in ("reason", "reasoning", "analysis", "explanation"):
            if key in obj and isinstance(obj[key], str):
                return obj[key][:500]

    # Fallback: last meaningful line
    lines = [l.strip() for l in text.splitlines() if l.strip() and len(l.strip()) > 5]
    return lines[-1][:500] if lines else "No analysis available"


VALID_KEYS = {
    "frame_rate", "reconstruction_backend",
    "fvdb_max_epochs", "fvdb_sh_degree", "fvdb_image_downsample_factor",
    "grut_n_iterations", "grut_render_method", "grut_strategy",
    "grut_downsample_factor", "splat_only_mode",
    "collision_mesh_enabled", "collision_mesh_method",
    "collision_mesh_target_faces", "collision_mesh_alpha", "collision_mesh_downsample",
}
GRUT_ONLY = {"grut_n_iterations", "grut_render_method", "grut_strategy", "grut_downsample_factor"}
FVDB_ONLY = {"fvdb_max_epochs", "fvdb_sh_degree", "fvdb_image_downsample_factor"}


def filter_for_backend(params: dict, current_backend: str) -> dict:
    if not current_backend:
        return params
    target = params.get("reconstruction_backend", current_backend)
    if target == "3dgrut":
        return {k: v for k, v in params.items() if k not in FVDB_ONLY}
    if target == "fvdb":
        return {k: v for k, v in params.items() if k not in GRUT_ONLY}
    return params


def extract_params(text: str, current_backend: str) -> str:
    cleaned = re.sub(r"`{3}[a-z]*\n?", "", text)
    for m in reversed(list(re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", cleaned))):
        try:
            obj = json.loads(m.group())
            if isinstance(obj.get("params"), dict):
                filtered = filter_for_backend(
                    {k: v for k, v in obj["params"].items() if k in VALID_KEYS},
                    current_backend,
                )
                if filtered:
                    return json.dumps(filtered)
            filtered = filter_for_backend(
                {k: v for k, v in obj.items() if k in VALID_KEYS},
                current_backend,
            )
            if filtered:
                return json.dumps(filtered)
        except Exception:
            continue
    return "{}"


def main():
    if len(sys.argv) < 2:
        print("Usage: extract_agent_output.py <verdict|reason|params> [backend]", file=sys.stderr)
        sys.exit(1)

    mode = sys.argv[1]
    raw = sys.stdin.read()
    text = unwrap_openclaw_payload(raw)

    if mode == "verdict":
        print(extract_verdict(text))
    elif mode == "reason":
        print(extract_reason(text))
    elif mode == "params":
        backend = sys.argv[2] if len(sys.argv) > 2 else ""
        print(extract_params(text, backend))
    else:
        print(f"Unknown mode: {mode}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
