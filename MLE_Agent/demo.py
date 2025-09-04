# demonstrates an agent using the tools # edit file and bash to run commands
try:
    from agents import Agent, Runner
except Exception:
    # Optional dependency not required for simple MCP tool demos
    Agent = Runner = None  # type: ignore
import os
from mcp_server import app as mcp_app, TOOLS, call_tool as server_call_tool
from pprint import pprint
import asyncio
from mcp import ClientSession, StdioServerParameters, stdio_client
import json

async def main():
    from openai import AsyncOpenAI
    
    #oss servers to use: mcp-server-filesystem

    client = AsyncOpenAI(api_key=os.environ['OPENROUTER_API_KEY'], base_url='https://openrouter.ai/api/v1')
    
    messages = [
        {
            "role": "system",
            "content": """
            You are given the following tools.
            - edit; to edit files
            - bash; run terminal commands
            """
        },
        {
            "role": "user",
            "content": f"from absolute path {os.getcwd()} write a tiny mlp demo in scratchpad.py that makes a forward pass and prints the output, then run via python in bash",
        },
        
    ]

    
    params = StdioServerParameters(command="python", args=["mcp_server.py"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            #tools = await session.list_tools()                    
            max_iters = 10
            iters = 0
            while iters < max_iters:
                
                response = await client.chat.completions.create(
                    model="moonshotai/kimi-k2:free",
                    messages=messages,
                    tools=[tool.to_tool_param() for tool in TOOLS]
                )
                print('RESPONSE:', response)
                
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
                        result = await session.call_tool(tool_name, tool_input)
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
    
    

""" 
"mcpServers": {
    "memory": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"]
    },
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/path/to/allowed/files"
      ]
    }
}
"""


async def test_client():
    params = StdioServerParameters(command="python", args=["mcp_server.py"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            for _ in range(10):
                result = await session.call_tool("bash", {"command": "ls -la"})
                print('result:', result)

if __name__ == "__main__":
    #asyncio.run(test_client())
    asyncio.run(main())