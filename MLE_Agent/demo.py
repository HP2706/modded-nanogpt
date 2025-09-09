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
from agent_viz import AgentLogger, LogLevel, Monitor, TokenUsage, Timing
from dataclasses import dataclass
from typing import Literal
from modal import Volume
from importlib import import_module
# Removed OpenAI SDK imports; using direct HTTP via httpx

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

def save_messages(msgs: list[dict], LOG_DIR: str) -> str:
    _ensure_logs_dir(LOG_DIR)
    ts = time.strftime("%Y%m%d-%H%M%S")
    fname = f"conversation_{ts}_{len(msgs)}.json"
    fpath = os.path.join(LOG_DIR, fname)
    payload = _sanitize_messages(msgs)
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return fpath

def from_logs(filename: str | None = None, LOG_DIR: str | None = None) -> list[dict]:
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


@dataclass
class StepLog:
    timing: Timing
    token_usage: TokenUsage | None = None
    

async def main(
    resume_from_conversation_path : str | None = None,
    resume_from_run_dir : str | None = None,
    file_name : Literal['modded_nanogpt_unoptimized.py', 'modded_nanogpt.py', 'cifar_speedrun_unoptimized.py', 'cifar_speedrun.py'] = 'cifar_speedrun_unoptimized.py',
    impl_paper : bool = False,
):
    if resume_from_run_dir is not None:
        os.environ["RUN_DIR"] = resume_from_run_dir
    os.environ["USE_MODAL_SANDBOX"] = "1"
    
    os.environ["FILE_NAME"] = file_name
    # first import mcp_server when env variables are set
    from mcp_server import mcp_app
    # if True, the workdir might already have changed modified-gpt, converted pdf to markdown etc
    # so we should change the prompt
    from utils import chat_completions_create
    
    #oss servers to use: mcp-server-filesystem

    # Using direct HTTP calls via httpx (no OpenAI SDK)

    LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
    SAVE_THRESHOLD = int(os.environ.get("DEMO_SAVE_THRESHOLD", "30"))

    mcp_client = Client(mcp_app)

    # Visualization helpers
    logger = AgentLogger(level=LogLevel.INFO)
    monitor = Monitor(tracked_model=None, logger=logger)

    async with mcp_client:
        tools = await mcp_client.list_tools()
        resume = os.environ.get("DEMO_RESUME", "0") == "1"

        if resume:
            loaded = from_logs(resume_from_conversation_path)
            assert loaded, "No logs found; starting fresh"
            logger.log_markdown("Resuming from latest logs", title="Resume")
            messages = loaded
            # set ephemeral cache control for all messages
            for msg_idx in range(4):
                # anthropic claude sonnet 4 has a limit of 4 messages with ephemeral cache control
                messages[msg_idx]['cache_control'] = {'type': 'ephemeral'}
        else:
            
            
            if resume_from_run_dir:
                vol = Volume.from_name("mle-sandbox", create_if_missing=True)
                mount_path = '/root/sandbox'
                
                resume_from_run_dir = resume_from_run_dir.replace(mount_path, '/root') # we need the local path
                remote_path = os.path.join(resume_from_run_dir, file_name)
                
                import tempfile
                with tempfile.TemporaryFile() as temp_file:
                    with vol.batch_download() as batch:
                        batch.get_file(remote_path, temp_file)
                    file_str = temp_file.read()
                
                
                FILE_TEXT = f"""
                You working dir might already contain a partial implementation of your task. The current implementation
                you have is:
                FILE_NAME : {file_name}
                __________
                {file_str}
                __________
                """
            else:
                path = os.path.join(os.path.dirname(__file__), 'environments', file_name)
                assert os.path.exists(path), f"File {path} does not exist"
                with open(path, 'r') as f:
                    file_str = f.read()
                
                FILE_TEXT = f"""
                In your working folder you have the following file:
                FILE_NAME : {file_name}
                __________
                {file_str}
                __________
                """
            
            maybe_pdf_to_markdown = '\n- pdf_to_markdown; curls a pdf from a url and converts it to easily readable markdown'
            messages = [
                {
                    "role": "system",
                    "content": f"""
                    You are given the following tools.
                    - edit; to edit files
                    - run_command; run terminal commands {maybe_pdf_to_markdown if impl_paper else ''}
                    """,
                    'cache_control': {'type': 'ephemeral'}
                }
            ]
            
            if impl_paper:
                messages.append({
                    "role": "user",
                    "content": f"""
                    modify the {file_name} file to implement the technique from this paper.
                    https://arxiv.org/pdf/2407.04620
                    Use the tools, note you can get the pdf to markdown using the pdf_to_markdown tool.
                    {FILE_TEXT}
                    You have access to an A100 80GB GPU, use it to conduct training runs.
                    """,
                    'cache_control': {'type': 'ephemeral'}
                })
            else:
                # here we do speedrunning, meaning we ask the model to 
                # implement a technique that speeds up training
                
                messages.append({
                    "role": "user",
                    "content": f"""
                    Implement a technique that speeds up the training of the model.
                    These techniques can be anything from changing the 
                    optimizer, data augmentation, learning rate schedule, 
                    architecture, etc.
                    
                    {FILE_TEXT}
                    """,
                    'cache_control': {
                        'type': 'ephemeral', 'ttl': "5m"} # TODO MAYBE CHANGE TO "1h"
                })
        
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
                response = await chat_completions_create(
                    #model="google/gemini-2.5-pro",
                    model="anthropic/claude-sonnet-4",
                    # model="openai/gpt-5",
                    messages=messages,
                    tools=tools_list,
                    reasoning={"max_tokens": 20_000}
                )
                t1 = time.time()

                # Token usage accounting (best-effort)

                usage = response.usage
                token_usage = TokenUsage(input_tokens=usage.prompt_tokens, output_tokens=usage.completion_tokens)

                step_log = StepLog(timing=Timing(start_time=t0, end_time=t1), token_usage=token_usage)
                monitor.update_metrics(step_log)
                logger.log(f"LLM call duration: {t1 - t0:.2f}s", level=LogLevel.INFO)

                reasoning = response.choices[0].message.reasoning
                if reasoning:
                    logger.log_markdown(reasoning, title="Reasoning Details")
                else:
                    print("No reasoning found in response")
                
                messages.append({
                    'role': response.choices[0].message.role,
                    'content': response.choices[0].message.content,
                    'tool_calls': response.choices[0].message.tool_calls,
                    'reasoning': reasoning,
                    })
                iters += 1

                if not saved_on_threshold and len(messages) >= SAVE_THRESHOLD:
                    try:
                        path = save_messages(messages, LOG_DIR)
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
                            content_blocks = result.content
                            out_texts = []
                            for b in content_blocks:
                                if b.type == 'text' and b.text:
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
                        content_blocks = result.content
                        out_texts = []
                        for b in content_blocks:
                            if b.type == 'text' and b.text:
                                out_texts.append(b.text)
                        messages.append({
                            'role': 'tool',
                            'tool_call_id': tool_call.id,
                            'content': "\n".join(out_texts) if out_texts else str(result),
                        })
        finally:
            try:
                if len(messages) >= SAVE_THRESHOLD:
                    path = save_messages(messages, LOG_DIR)
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
    #
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_from_conversation_path", type=str, default=None)
    parser.add_argument("--resume_from_run_dir", type=str, default=None)
    parser.add_argument("--file_name", type=str, default='cifar_speedrun_unoptimized.py')
    parser.add_argument("--impl_paper", type=bool, default=False)
    args = parser.parse_args()
    asyncio.run(main(
        args.resume_from_conversation_path, 
        args.resume_from_run_dir, 
        args.file_name, 
        args.impl_paper
    ))