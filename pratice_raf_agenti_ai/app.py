import asyncio
from typing import Annotated, TypedDict

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from langchain_mcp_adapters.client import MultiServerMCPClient


import sys

load_dotenv()


# ==========================================
# 1. Connect to MCP Server
# ==========================================

client = MultiServerMCPClient(
    {
        "calculator": { # name/identifier you give to this MCP server connection.
            "command": sys.executable,
            "args": ["server.py"],
            "transport": "stdio",
        }
    }
)


# ==========================================
# 2. State Definition
# ==========================================

class State(TypedDict):

    messages: Annotated[list, add_messages]


# ==========================================
# 3. Main
# ==========================================

async def main():

    # Get tools from MCP server wait until the MCP server responds with the tools
    tools = await client.get_tools() 

    print("Available MCP tools:")

    for tool in tools:
        print("-", tool.name)


    # ======================================
    # 4. LLM
    # ======================================

    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-lite-latest",
        temperature=0.7
    )

    # Give MCP tools to LLM
    llm_with_tools = llm.bind_tools(tools)


    # ======================================
    # 5. LLM Node
    # ======================================

    def llm_node(state: State):

        response = llm_with_tools.invoke(
            state["messages"]
        )

        return {
            "messages": [response]
        }


    # ======================================
    # 6. Tool Node
    # ======================================

    tool_node = ToolNode(tools)


    # ======================================
    # 7. Routing
    # ======================================

    def should_cont(state: State):

        last_msg = state["messages"][-1]

        print("\nLLM RESPONSE:")
        print(last_msg)

        if last_msg.tool_calls:
            return "tools"

        return END


    # ======================================
    # 8. Build Graph
    # ======================================

    graph = StateGraph(State)

    graph.add_node("llm", llm_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(
        START,
        "llm"
    )

    graph.add_conditional_edges(
        "llm",
        should_cont,
        {
            "tools": "tools",
            END: END
        }
    )

    graph.add_edge(
        "tools",
        "llm"
    )


    app = graph.compile()


    # ======================================
#     # 9. Run Method	Meaning
# invoke()	Run graph synchronously
# ainvoke()	Run graph asynchronously
    # ======================================

    result = await app.ainvoke({
        "messages": [
            (
                "user",
                "What is 25 + 17?"
            )
        ]
    })


    # ======================================
    # 10. Final Answer
    # ======================================

    print("\nFINAL ANSWER:")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())