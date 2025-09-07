from pprint import pprint

def _to_dumpable(res):
    try:
        if hasattr(res, "model_dump") and callable(res.model_dump):
            return res.model_dump()
        if hasattr(res, "dict") and callable(res.dict):
            return res.dict()
        if isinstance(res, dict):
            return res
        # Fallback: extract common fields defensively
        return {
            "content": getattr(res, "content", None),
            "isError": getattr(res, "isError", False) or getattr(res, "is_error", False),
            "meta": getattr(res, "meta", None),
        }
    except Exception:
        return {"repr": repr(res)}

def format_print_tool(tool_name: str, tool_args: dict, res) -> None:
    data = {
        'tool_name': tool_name,
        'tool_args': tool_args,
        **_to_dumpable(res),
    }
    pprint(data)
