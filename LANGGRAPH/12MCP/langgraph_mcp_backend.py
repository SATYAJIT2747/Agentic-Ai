# ============================================================
# IMPORTS
# ============================================================

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated

# Message types used by LangGraph
from langchain_core.messages import BaseMessage

# Gemini LLM
from langchain_google_genai import ChatGoogleGenerativeAI

# Async SQLite checkpointer for storing conversation state
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# add_messages automatically appends new messages to the state
from langgraph.graph.message import add_messages

# ToolNode executes tools requested by the LLM
# tools_condition decides whether the LLM wants to call a tool or finish
from langgraph.prebuilt import ToolNode, tools_condition

# Normal LangChain tool: DuckDuckGo search
from langchain_community.tools import DuckDuckGoSearchRun

# BaseTool is used for type hinting MCP tools
from langchain_core.tools import BaseTool

# MCP client which connects LangChain/LangGraph to MCP servers
from langchain_mcp_adapters.client import MultiServerMCPClient

# Load environment variables from .env
from dotenv import load_dotenv

# Async SQLite
import aiosqlite

# Python's async programming library
import asyncio

# Used to run a dedicated async event loop in another thread
import threading


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# ASYNC EVENT LOOP
# ============================================================

# Create a new asyncio event loop.
#
# An event loop is responsible for running async functions
# and handling operations that are waiting for I/O.
#
# For example:
#
#     await LLM API call
#            ↓
#     waiting for response
#            ↓
#     event loop can handle other async work
#
_ASYNC_LOOP = asyncio.new_event_loop()


# Create a separate thread to continuously run our
# dedicated async event loop.
_ASYNC_THREAD = threading.Thread(
    target=_ASYNC_LOOP.run_forever,
    daemon=True
)

_ASYNC_THREAD.start()


# ============================================================
# ASYNC HELPER FUNCTIONS
# ============================================================

def _submit_async(coro):
    """
    Submit a coroutine to our dedicated async event loop.

    A coroutine is an async function that has not yet
    finished executing.
    """

    return asyncio.run_coroutine_threadsafe(
        coro,
        _ASYNC_LOOP
    )


def run_async(coro):
    """
    Run an async function and WAIT for its result.

    We use this when normal/synchronous code needs
    the result of an async operation.
    """

    return _submit_async(coro).result()


def submit_async_task(coro):
    """
    Schedule an async task on the background event loop.

    Unlike run_async(), this does not wait for the result.
    """

    return _submit_async(coro)


# ============================================================
# 1. LLM
# ============================================================

# We are using GOOGLE GEMINI, NOT OpenAI.
#
# This is our main LLM.
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-lite-latest"
)


# ============================================================
# 2. NORMAL LANGCHAIN TOOL
# ============================================================

# DuckDuckGo is a normal LangChain tool.
#
# It is NOT coming from MCP.
#
# The LLM can use this tool when it needs
# information from the web.
search_tool = DuckDuckGoSearchRun(
    region="us-en"
)


# ============================================================
# 3. MCP CLIENT
# ============================================================

# MultiServerMCPClient allows our LangGraph application
# to connect to one or more MCP servers.
#
# Here we connect to our own calculator MCP server.
#
# IMPORTANT:
#
#     LangGraph
#         ↓
#     MCP Client
#         ↓
#     calculator MCP server
#         ↓
#     calculator tools
#
client = MultiServerMCPClient({

    # "calculator" is simply the name we give
    # to this MCP server.
    "calculator": {

        # stdio means:
        #
        # The MCP client starts the server as a
        # local process and communicates with it
        # using standard input/output.
        #
        # Client ──stdin/stdout──> MCP Server
        #
        "transport": "stdio",

        # Use Python to start the MCP server.
        "command": "python",

        # This is the Python file containing
        # our MCP server.
        #
        # Equivalent to running:
        #
        #     python mcp_server.py
        #
        "args": ["mcp_server.py"]
    }
})


# ============================================================
# 4. GET TOOLS FROM MCP SERVER
# ============================================================

def load_mcp_tools() -> list[BaseTool]:
    """
    Ask the MCP server what tools it provides.

    Our calculator MCP server might expose:

        add
        subtract
        multiply
        divide

    get_tools() retrieves those tools so that
    LangGraph/LangChain can give them to Gemini.
    """

    try:

        # get_tools() is asynchronous.
        #
        # Therefore we use run_async() to execute it
        # on our async event loop and get the result.
        return run_async(
            client.get_tools()
        )

    except Exception as e:

        # If the MCP server fails to start/connect,
        # print the error and return an empty list.
        print("MCP ERROR:", e)

        return []


# Actually retrieve the tools from the MCP server.
mcp_tools = load_mcp_tools()


# ============================================================
# 5. COMBINE ALL TOOLS
# ============================================================

# We now have TWO sources of tools:
#
# 1. search_tool
#       ↓
#    Normal LangChain tool
#
# 2. mcp_tools
#       ↓
#    Tools coming from our MCP server
#
# *mcp_tools means:
# "put every tool inside mcp_tools into this list"
#
tools = [
    search_tool,
    *mcp_tools
]


# Print the tools so we can see what the application
# actually received from the MCP server.
#
# Example:
#
# TOOLS: ['duckduckgo_search', 'add', 'subtract',
#         'multiply', 'divide']
#
print(
    "TOOLS:",
    [tool.name for tool in tools]
)


# ============================================================
# 6. GIVE TOOLS TO GEMINI
# ============================================================

# bind_tools() tells Gemini:
#
# "These are the tools you are allowed to use."
#
# Gemini can now decide:
#
# User: "What is 25 * 4?"
#
# Gemini
#    ↓
# chooses multiply tool
#    ↓
# MCP calculator executes it
#
llm_with_tools = llm.bind_tools(tools)


# ============================================================
# 7. LANGGRAPH STATE
# ============================================================

class ChatState(TypedDict):

    # This stores the conversation messages.
    #
    # add_messages tells LangGraph to APPEND new
    # messages instead of replacing the old messages.
    messages: Annotated[
        list[BaseMessage],
        add_messages
    ]


# ============================================================
# 8. CHAT NODE
# ============================================================

async def chat_node(state: ChatState):
    """
    This is the LLM node in our LangGraph.

    It receives the current conversation,
    sends it to Gemini,
    and returns Gemini's response.
    """

    # Get all previous messages from the state.
    messages = state["messages"]


    # IMPORTANT ASYNC PART
    #
    # ainvoke() is the asynchronous version of invoke().
    #
    # Gemini API requires network communication.
    #
    # So instead of blocking while waiting for Gemini,
    # we use await.
    #
    #     send request
    #          ↓
    #        await
    #          ↓
    #     Gemini processes
    #          ↓
    #     response comes back
    #
    response = await llm_with_tools.ainvoke(
        messages
    )


    # Add Gemini's response to the state.
    return {
        "messages": [response]
    }


# ============================================================
# 9. TOOL NODE
# ============================================================

# ToolNode is responsible for actually EXECUTING
# the tool requested by Gemini.
#
# Example:
#
# Gemini says:
#
#     call multiply(a=10, b=5)
#
# ToolNode finds the multiply tool and executes it.
#
tool_node = ToolNode(tools)


# ============================================================
# 10. SQLITE CHECKPOINTER
# ============================================================

async def _init_checkpointer():
    """
    Create an asynchronous SQLite connection.

    This allows LangGraph to save conversation state.
    """

    conn = await aiosqlite.connect(
        database="chatbot.db"
    )

    return AsyncSqliteSaver(conn)


# _init_checkpointer() is async.
#
# But this code is currently outside an async function.
#
# Therefore we use run_async() to execute it.
checkpointer = run_async(
    _init_checkpointer()
)


# ============================================================
# 11. CREATE LANGGRAPH
# ============================================================

graph = StateGraph(ChatState)


# Add our LLM node.
graph.add_node(
    "chat_node",
    chat_node
)


# Add our tool node.
graph.add_node(
    "tools",
    tool_node
)


# ============================================================
# GRAPH FLOW
# ============================================================

# START
#   ↓
# chat_node
#
graph.add_edge(
    START,
    "chat_node"
)


# After Gemini responds, tools_condition checks:
#
# Did Gemini request a tool?
#
# YES → tools
#
# NO  → END
#
graph.add_conditional_edges(
    "chat_node",
    tools_condition
)


# After the tool finishes:
#
# tools
#   ↓
# chat_node
#
# We send the tool result back to Gemini so that
# Gemini can generate the final answer.
graph.add_edge(
    "tools",
    "chat_node"
)


# ============================================================
# GRAPH VISUALIZATION
# ============================================================
#
# The complete flow is:
#
#
#                  ┌─────────────┐
#                  │ chat_node   │
#                  │   Gemini    │
#                  └──────┬──────┘
#                         │
#                  tools_condition
#                    /           \
#                   /             \
#              Tool needed       No tool
#                 ↓                 ↓
#          ┌─────────────┐        END
#          │    tools    │
#          └──────┬──────┘
#                 │
#                 ↓
#            chat_node
#
#
# If the tool is an MCP tool:
#
# chat_node
#     ↓
# Gemini requests calculator
#     ↓
# ToolNode
#     ↓
# MCP tool
#     ↓
# Calculator MCP server
#     ↓
# result
#     ↓
# chat_node
#     ↓
# Gemini final answer
#
# ============================================================


# ============================================================
# 12. COMPILE GRAPH
# ============================================================

# Compile the graph and attach the SQLite checkpointer.
#
# The checkpointer allows LangGraph to save state,
# which is useful for maintaining conversations/threads.
chatbot = graph.compile(
    checkpointer=checkpointer
)


# ============================================================
# 13. GET ALL SAVED THREADS
# ============================================================

async def _alist_threads():
    """
    Retrieve all conversation thread IDs
    stored in SQLite.
    """

    all_threads = set()


    # alist() asynchronously goes through
    # saved checkpoints.
    async for checkpoint in checkpointer.alist(None):

        # Extract the thread ID from the checkpoint.
        all_threads.add(
            checkpoint.config[
                "configurable"
            ]["thread_id"]
        )


    return list(all_threads)


def retrieve_all_threads():
    """
    Normal/synchronous wrapper around
    the asynchronous _alist_threads() function.
    """

    return run_async(
        _alist_threads()
    )