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

async def main():
    from openai import AsyncOpenAI
    
    #oss servers to use: mcp-server-filesystem

    client = AsyncOpenAI(api_key=os.environ['OPENROUTER_API_KEY'], base_url='https://openrouter.ai/api/v1')
    
    os.environ["USE_MODAL_SANDBOX"] = "1"
    mcp_client = Client(mcp_app)
    
    async with mcp_client:
        tools = await mcp_client.list_tools()
        
        output = await mcp_client.call_tool("bash", {"command": "ls -la"})
        pprint(output)        
        
        messages = [
            {
                "role": "system",
                "content": """
                You are given the following tools.
                - edit; to edit files
                - bash; run terminal commands
                - pdf_to_markdown; curls a pdf from a url and converts it to easily readable markdown
                """
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
            },
        ]

        #ChatCompletionToolUnionParam
        from openai.types.chat.chat_completion_function_tool_param import ChatCompletionFunctionToolParam
        from openai.types.shared_params.function_definition import FunctionDefinition
        max_iters = 10
        iters = 0
        while iters < max_iters:
            
            t0 = time.time()
            response = await client.chat.completions.create(
                model="anthropic/claude-sonnet-4",
                messages=messages,
                tools=[
                    ChatCompletionFunctionToolParam(
                        type="function", 
                        function=FunctionDefinition(
                            name=tool.name, description=tool.description, parameters=tool.inputSchema)
                        ) 
                    for tool in tools
                ]
            )
            t1 = time.time()
            print(f"Time taken: {t1 - t0} seconds")
            
            messages.append({
                'role': response.choices[0].message.role,
                'content': response.choices[0].message.content
            })
            iters += 1
            pprint(messages[-1])
            
            if response.choices[0].message.tool_calls:
                print("number of tool calls:", len(response.choices[0].message.tool_calls))
                for tool_call in response.choices[0].message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_input = json.loads(tool_call.function.arguments)
                    # call mcp server tool
                    result = await mcp_client.call_tool(tool_name, tool_input)
                    pprint(f"result: for tool call {tool_call}, {result}")
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





if __name__ == "__main__":
    asyncio.run(main())