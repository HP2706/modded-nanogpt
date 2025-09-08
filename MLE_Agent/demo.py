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

async def main():
    from openai import AsyncOpenAI
    
    #oss servers to use: mcp-server-filesystem

    client = AsyncOpenAI(api_key=os.environ['OPENROUTER_API_KEY'], base_url='https://openrouter.ai/api/v1')

    os.environ["USE_MODAL_SANDBOX"] = "1"
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

        output = await mcp_client.call_tool("bash", {"command": "ls -la"})
        logger.log_markdown("Bootstrapping with `ls -la` via bash tool:", title="Bootstrap", level=LogLevel.INFO)
        try:
            content_blocks = getattr(output, 'content', [])
            out_texts = []
            for b in content_blocks:
                if getattr(b, 'type', '') == 'text' and hasattr(b, 'text'):
                    out_texts.append(b.text)
            if out_texts:
                logger.log_markdown("\n".join(out_texts[:1]), title="bash result (truncated)")
            else:
                logger.log_markdown(str(output), title="bash result (raw)")
        except Exception as e:
            logger.log_error(f"Error rendering bootstrap output: {e}")
        
        messages = [
            {
                "role": "system",
                "content": """
                You are given the following tools.
                - edit; to edit files
                - bash; run terminal commands
                - pdf_to_markdown; curls a pdf from a url and converts it to easily readable markdown
                """,
                'cache_control': {
                    'type': 'ephemeral'
                }
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
                'cache_control': {
                    'type': 'ephemeral'
                }
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
        while iters < max_iters:
            logger.log_rule(f"Iteration {iters+1}")      
            tools_list=[
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                }
                for tool in tools
            ]
            
            
            t0 = time.time()
            response = await client.chat.completions.create(
                #model="google/gemini-2.5-pro",
                model="anthropic/claude-sonnet-4",
                #model="openai/gpt-5",
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
                'tool_calls': response.choices[0].message.tool_calls
            })
            iters += 1
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
                    tool_input = json.loads(tool_call.function.arguments)
                    logger.log_code("Tool Invocation", json.dumps({"name": tool_name, "args": tool_input}, indent=2))
                    try:
                        # call mcp server tool
                        result = await mcp_client.call_tool(tool_name, tool_input)
                        # Render tool result
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
                            is_error=True
                        )
                        
                    # Collect text blocks
                    content_blocks = getattr(result, 'content', [])
                    out_texts = []
                    for b in content_blocks:
                        if getattr(b, 'type', '') == 'text' and hasattr(b, 'text'):
                            out_texts.append(b.text)
                    
                    messages.append({
                        'role': 'tool',
                        'tool_call_id': tool_call.id,
                        'content': "\n".join(out_texts) if out_texts else str(result)
                    })

        # Final totals
        totals = monitor.get_total_token_counts()
        logger.log_markdown(
            json.dumps(totals.dict(), indent=2),
            title="Total Token Usage",
        )





if __name__ == "__main__":
    asyncio.run(main())
