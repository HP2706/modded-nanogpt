# demonstrates an agent using the tools # edit file and bash to run commands
try:
    from agents import Agent, Runner
except Exception:
    # Optional dependency not required for simple MCP tool demos
    Agent = Runner = None  # type: ignore
import os
from pprint import pprint
import asyncio
from fastmcp import Client
import json
import time
from mcp_server import mcp_app
from agent_viz import AgentLogger, LogLevel, Monitor, TokenUsage, Timing
from dataclasses import dataclass
from openai.types.chat.chat_completion_function_tool_param import ChatCompletionFunctionToolParam

async def main(
    resume_from_path : str | None = None,
):
    from openai import AsyncOpenAI
    
    #oss servers to use: mcp-server-filesystem

    client = AsyncOpenAI(api_key=os.environ['OPENROUTER_API_KEY'], base_url='https://openrouter.ai/api/v1')

    LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
    SAVE_THRESHOLD = int(os.environ.get("DEMO_SAVE_THRESHOLD", "30"))

    def _ensure_logs_dir(path: str) -> None:
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)

    def _serialize_tool_call(tc: object) -> dict:
        # Accept dict or SDK object, return JSON-friendly dict
        if isinstance(tc, dict):
            out = {
                "id": tc.get("id"),
                "type": tc.get("type", "function"),
                "function": None,
            }
            fn = tc.get("function")
            if isinstance(fn, dict):
                args = fn.get("arguments")
                if not isinstance(args, (str, type(None))):
                    try:
                        args = json.dumps(args)
                    except Exception:
                        args = str(args)
                out["function"] = {"name": fn.get("name"), "arguments": args}
            return out
        # Fallback for SDK objects
        fn = getattr(tc, "function", None)
        name = getattr(fn, "name", None) if fn is not None else None
        args = getattr(fn, "arguments", None) if fn is not None else None
        if not isinstance(args, (str, type(None))):
            try:
                args = json.dumps(args)
            except Exception:
                args = str(args)
        return {
            "id": getattr(tc, "id", None),
            "type": getattr(tc, "type", "function"),
            "function": {"name": name, "arguments": args},
        }

    def _sanitize_messages(msgs: list[dict]) -> list[dict]:
        safe: list[dict] = []
        for m in msgs:
            d = {k: v for k, v in m.items() if k in ("role", "content", "name", "tool_call_id", "tool_calls", "cache_control")}
            # Ensure content is JSON-serializable
            c = d.get("content")
            if c is None or isinstance(c, (str, int, float, bool, list, dict)):
                pass
            else:
                d["content"] = str(c)
            # Serialize tool_calls if present
            if "tool_calls" in m and m["tool_calls"] is not None:
                tcs = m["tool_calls"]
                if isinstance(tcs, list):
                    d["tool_calls"] = [_serialize_tool_call(tc) for tc in tcs]
                else:
                    d["tool_calls"] = [_serialize_tool_call(tcs)]
            safe.append(d)
        return safe

    def save_messages(msgs: list[dict]) -> str:
        _ensure_logs_dir(LOG_DIR)
        ts = time.strftime("%Y%m%d-%H%M%S")
        fname = f"conversation_{ts}_{len(msgs)}.json"
        fpath = os.path.join(LOG_DIR, fname)
        payload = _sanitize_messages(msgs)
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return fpath

    def from_logs(filename: str | None = None) -> list[dict]:
        _ensure_logs_dir(LOG_DIR)
        target = None
        if filename:
            target = os.path.join(LOG_DIR, filename)
        else:
            files = [
                os.path.join(LOG_DIR, f)
                for f in os.listdir(LOG_DIR)
                if f.endswith('.json') and os.path.isfile(os.path.join(LOG_DIR, f))
            ]
            if files:
                target = max(files, key=lambda p: os.path.getmtime(p))
        if not target or not os.path.exists(target):
            return []
        try:
            with open(target, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "messages" in data:
                return data["messages"]
            if isinstance(data, list):
                return data
        except Exception:
            return []
        return []

    os.environ["USE_MODAL_SANDBOX"] = "1"
    os.environ["RUN_DIR"] = "/root/sandbox/runs/2025-09-08_19-09-54" #TODO remove
    mcp_client = Client(mcp_app)

    # Visualization helpers
    logger = AgentLogger(level=LogLevel.INFO)
    monitor = Monitor(tracked_model=None, logger=logger)

    @dataclass
    class StepLog:
        timing: Timing
        token_usage: TokenUsage | None = None
    
    async with mcp_client:
        tools = await mcp_client.list_tools()
        resume = os.environ.get("DEMO_RESUME", "0") == "1"
        messages: list[dict]
        if resume:
            loaded = from_logs(resume_from_path)
            if loaded:
                logger.log_markdown("Resuming from latest logs", title="Resume")
                messages = loaded
            else:
                logger.log_markdown("No logs found; starting fresh", title="Resume")
                messages = [
                    {
                        "role": "system",
                        "content": """
                        You are given the following tools.
                        - edit; to edit files
                        - run_command; run terminal commands
                        - pdf_to_markdown; curls a pdf from a url and converts it to easily readable markdown
                        """,
                        'cache_control': {'type': 'ephemeral'}
                    },
                    {
                        "role": "user",
                        "content": """
                        modify the modded-nanogpt.py file to implement the technique from this paper.
                        https://arxiv.org/pdf/2407.04620 .
                        But check first if the paper hasnt already been downloaded in the papers folder.
                        Use the tools, note you can get the pdf to markdown using the pdf_to_markdown tool.
                        In your working folder(which you can see via bash), i have a minimal file for training a
                        gpt2 style model. modify this file to implement the paper. 
                        """,
                        'cache_control': {'type': 'ephemeral'}
                    },
                ]
        else:
            messages = [
                {
                    "role": "system",
                    "content": """
                    You are given the following tools.
                    - edit; to edit files
                    - run_command; run terminal commands
                    - pdf_to_markdown; curls a pdf from a url and converts it to easily readable markdown
                    """,
                    'cache_control': {'type': 'ephemeral'}
                },
                {
                    "role": "user",
                    "content": """
                    modify the modded-nanogpt.py file to implement the technique from this paper.
                    https://arxiv.org/pdf/2407.04620
                    Use the tools, note you can get the pdf to markdown using the pdf_to_markdown tool.
                    In your working folder(which you can see via bash), i have a minimal file for training a
                    gpt2 style model. modify this file to implement the paper.
                    """,
                    'cache_control': {'type': 'ephemeral'}
                },
            ]
        
        # Log initial run context
        for msg in messages:
            logger.log_task(
                content=(
                    msg['content']
                ),
                subtitle=f"Tools available: {', '.join([t.name for t in tools])}",
                title="Demo Run",
            )
        
            
        
        max_iters = 100
        iters = 0
        saved_on_threshold = False
        try:
            while iters < max_iters:
                logger.log_rule(f"Iteration {iters+1}")
                tools_list = [
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.inputSchema,
                        },
                    }
                    for tool in tools
                ]

                t0 = time.time()
                response = await client.chat.completions.create(
                    #model="google/gemini-2.5-pro",
                    model="anthropic/claude-sonnet-4",
                    # model="openai/gpt-5",
                    messages=messages,
                    tools=tools_list,
                )
                t1 = time.time()

                # Token usage accounting (best-effort)
                usage = getattr(response, 'usage', None)
                token_usage = None
                if usage is not None:
                    try:
                        input_tokens = getattr(usage, 'prompt_tokens', None) or getattr(usage, 'total_tokens', 0)
                        output_tokens = getattr(usage, 'completion_tokens', 0)
                        token_usage = TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)
                    except Exception:
                        token_usage = None

                step_log = StepLog(timing=Timing(start_time=t0, end_time=t1), token_usage=token_usage)
                monitor.update_metrics(step_log)
                logger.log(f"LLM call duration: {t1 - t0:.2f}s", level=LogLevel.INFO)

                messages.append({
                    'role': response.choices[0].message.role,
                    'content': response.choices[0].message.content,
                    'tool_calls': response.choices[0].message.tool_calls,
                })
                iters += 1

                if not saved_on_threshold and len(messages) >= SAVE_THRESHOLD:
                    try:
                        path = save_messages(messages)
                        logger.log_markdown(f"Saved to {path}", title="Log Checkpoint")
                        saved_on_threshold = True
                    except Exception as e:
                        logger.log_error(f"Error saving logs: {e}")

                # Show assistant reply summary
                try:
                    assistant_content = response.choices[0].message.content or ""
                    if isinstance(assistant_content, str):
                        preview = assistant_content.strip()
                    else:
                        preview = str(assistant_content)
                    if len(preview) > 1500:
                        preview = preview[:1500] + "\n... (truncated)"
                    logger.log_markdown(preview, title="Assistant Reply")
                except Exception as e:
                    logger.log_error(f"Error rendering assistant reply: {e}")

                if response.choices[0].message.tool_calls:
                    logger.log_markdown(
                        f"number of tool calls: {len(response.choices[0].message.tool_calls)}",
                        title="Tool Calls",
                    )
                    for tool_call in response.choices[0].message.tool_calls:
                        tool_name = tool_call.function.name
                        try:
                            tool_input = json.loads(tool_call.function.arguments)
                        except Exception as e:
                            logger.log_error(f"Error parsing tool input: {e}")
                            messages.append({
                                'role': 'tool',
                                'tool_call_id': tool_call.id,
                                'content': f"Error parsing tool input: {e}",
                            })
                            continue
                        logger.log_code("Tool Invocation", json.dumps({"name": tool_name, "args": tool_input}, indent=2))
                        try:
                            result = await mcp_client.call_tool(tool_name, tool_input)
                            print(f"result: {result} from tool {tool_name}")
                            content_blocks = getattr(result, 'content', [])
                            out_texts = []
                            for b in content_blocks:
                                if getattr(b, 'type', '') == 'text' and hasattr(b, 'text'):
                                    out_texts.append(b.text)
                            rendered = "\n".join(out_texts) if out_texts else str(result)
                            if len(rendered) > 2000:
                                rendered = rendered[:2000] + "\n... (truncated)"
                            logger.log_markdown(rendered, title=f"{tool_name} result")
                        except Exception as e:
                            logger.log_error(f"Error rendering tool result: {str(e)}")
                            from mcp.types import TextContent, CallToolResult
                            result = CallToolResult(
                                content=[TextContent(type="text", text=f"Error calling tool: {str(e)}")],
                                is_error=True,
                            )
                        content_blocks = getattr(result, 'content', [])
                        out_texts = []
                        for b in content_blocks:
                            if getattr(b, 'type', '') == 'text' and hasattr(b, 'text'):
                                out_texts.append(b.text)
                        messages.append({
                            'role': 'tool',
                            'tool_call_id': tool_call.id,
                            'content': "\n".join(out_texts) if out_texts else str(result),
                        })
        finally:
            try:
                if len(messages) >= SAVE_THRESHOLD:
                    path = save_messages(messages)
                    logger.log_markdown(f"Saved to {path}", title="Final Log Save")
            except Exception as e:
                logger.log_error(f"Error saving logs in finally: {e}")
                # pickle them
                import pickle
                with open("messages.pkl", "wb") as f:
                    pickle.dump(messages, f)

        # Final totals
        totals = monitor.get_total_token_counts()
        logger.log_markdown(
            json.dumps(totals.dict(), indent=2),
            title="Total Token Usage",
        )

if __name__ == "__main__":
    asyncio.run(main())