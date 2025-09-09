import os
from types import SimpleNamespace
from typing import Any, Optional

import httpx


def _to_namespace(obj: Any) -> Any:
    """Recursively convert dicts to SimpleNamespace for attr-style access.

    Lists are converted element-wise. Scalars are returned as-is.
    """
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert objects to JSON-serializable structures.

    - SimpleNamespace -> dict
    - dict -> dict (values converted)
    - list/tuple -> list
    - others unchanged
    Also normalizes tool_calls.function.arguments to string when present.
    """
    if isinstance(obj, SimpleNamespace):
        return _to_jsonable(vars(obj))
    if isinstance(obj, dict):
        out = {k: _to_jsonable(v) for k, v in obj.items()}
        # Normalize tool_calls argument shape if present
        try:
            if "tool_calls" in out and isinstance(out["tool_calls"], list):
                norm_calls = []
                for tc in out["tool_calls"]:
                    if isinstance(tc, SimpleNamespace):
                        tc = _to_jsonable(tc)
                    if isinstance(tc, dict):
                        fn = tc.get("function")
                        if isinstance(fn, SimpleNamespace):
                            fn = _to_jsonable(fn)
                        if isinstance(fn, dict):
                            args = fn.get("arguments")
                            if not isinstance(args, (str, type(None))):
                                import json as _json
                                try:
                                    fn["arguments"] = _json.dumps(args)
                                except Exception:
                                    fn["arguments"] = str(args)
                            tc["function"] = fn
                    norm_calls.append(tc)
                out["tool_calls"] = norm_calls
        except Exception:
            pass
        return out
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj

async def chat_completions_create(
    *,
    messages: list[dict],
    model: str,
    tools: Optional[list[dict]] = None,
    reasoning: Optional[dict] = None,
    api_key: Optional[str] = None,
    base_url: str = "https://openrouter.ai/api/v1",
    extra_headers: Optional[dict] = None,
    timeout: float = 60.0,
) -> Any:
    """Call OpenAI-compatible chat completions over HTTP using httpx (async).

    Returns an object allowing attribute access like response.choices[0].message.content
    to match the usage pattern in demo.py.
    """
    if api_key is None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set and no api_key provided")

    url = f"{base_url.rstrip('/')}/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)

    payload: dict[str, Any] = {
        "model": model,
        "messages": _to_jsonable(messages),
    }
    if tools:
        payload["tools"] = _to_jsonable(tools)
    if reasoning:
        payload["reasoning"] = _to_jsonable(reasoning)

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, headers=headers, json=payload)
        # Raise for non-2xx with helpful message
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            detail = None
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            raise RuntimeError(f"Chat completion request failed: {e} | Detail: {detail}")

        data = resp.json()
        return _to_namespace(data)
