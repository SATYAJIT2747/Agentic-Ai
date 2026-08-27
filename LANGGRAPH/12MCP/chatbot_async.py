from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
load_dotenv()

import os

os.environ["LANGCHAIN_PROJECT"] = "chat-bot-Tools"


# -------------------
# LLM
# -------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-flash-lite-latest"
)
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient({
    "calculator": {
        "transport": "stdio",
        "command": "python",
        "args": ["mcp_server.py"]
    },

    "weather": {
        "transport": "stdio",
        "command": "npx",
        "args": [
            "-y",
            "@dangahagan/weather-mcp@latest"
        ]
    }
})

# -------------------
# Tools
# -------------------






# -------------------
# State
# -------------------

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -------------------
# Build Graph
# -------------------
# async def task()
#       ↓
# "This function may have to WAIT for something"
# If I want to use `await`
#         ↓
# my function needs to be `async`
# await asyncio.sleep(3)

# means:

# "I'm going to wait 3 seconds here. While I'm waiting, Python's async system can do other work."
async def build_graph():
    
    tools = await client.get_tools()
    llm_with_tools = llm.bind_tools(tools)
    async def chat_node(state: ChatState):

        messages = state["messages"]

        # IMPORTANT: ainvoke() is async, so use await
        response = await llm_with_tools.ainvoke(messages)

        print("TOOL CALLS:", response.tool_calls)

        return {
            "messages": [response]
        }

    # Tool node
    tool_node = ToolNode(tools)

    # Create graph
    graph = StateGraph(ChatState)

    # Add nodes
    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)

    # START → chat
    graph.add_edge(START, "chat_node")

    # chat → tools OR END
    graph.add_conditional_edges(
        "chat_node",
        tools_condition
    )

    # tools → chat
    graph.add_edge("tools", "chat_node")

    # Compile
    chatbot = graph.compile()

    return chatbot


# -------------------
# Main
# -------------------

async def main():

    chatbot = await build_graph()

    result = await chatbot.ainvoke(
        {
            "messages": [
                HumanMessage(content="weathre here at iiit hydrabad today")
            ]
        }
    )

    print(result["messages"][-1].content)
# chatbot.ainvoke() starts the whole workflow.

# main()
#   │
#   │ chatbot.ainvoke()
#   ▼
# ┌─────────────────────────┐
# │      LangGraph          │
# │                         │
# │ START → chat_node       │
# │            ↓            │
# │          tools          │
# │            ↓            │
# │         chat_node       │
# │            ↓            │
# │           END           │
# └─────────────────────────┘

# -------------------
# Run
# -------------------

if __name__ == "__main__":
    asyncio.run(main())
    
# Synchronous
# response = llm.invoke(messages)

# Python:

# Send request
#      ↓
# WAIT ⏳
#      ↓
# Response
#      ↓
# Continue
# Asynchronous
# response = await llm.ainvoke(messages)

# Python:

# Send request
#      ↓
# This function waits ⏳
#      ↓
# Other async work can run
#      ↓
# Response comes
#      ↓
# Continue this function

# That's the main purpose of async/await.