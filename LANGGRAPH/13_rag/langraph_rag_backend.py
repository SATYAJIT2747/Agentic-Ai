from __future__ import annotations

# ============================================================
# IMPORTS
# ============================================================

import os
import sqlite3
import tempfile

from typing import Annotated, Any, Dict, Optional, TypedDict

from dotenv import load_dotenv

# PDF loading
from langchain_community.document_loaders import PyPDFLoader

# Text splitting
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vector store
from langchain_community.vectorstores import FAISS

# DuckDuckGo search
from langchain_community.tools import DuckDuckGoSearchRun

# LangChain messages
from langchain_core.messages import (
    BaseMessage,
    SystemMessage
)

# LangChain tools
from langchain_core.tools import tool

# ============================================================
# GEMINI
# ============================================================

# Gemini chat model
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

# LangGraph
from langgraph.checkpoint.sqlite import SqliteSaver

from langgraph.graph import (
    START,
    StateGraph
)

from langgraph.graph.message import add_messages

from langgraph.prebuilt import (
    ToolNode,
    tools_condition
)


# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================

# Load variables from .env
#
# Your .env should contain something like:
#
# GOOGLE_API_KEY=your_gemini_api_key
#
load_dotenv()


# ============================================================
# 1. GEMINI LLM + EMBEDDINGS
# ============================================================

# Gemini LLM
#
# This replaces:
#
#     ChatOpenAI(model="gpt-4o-mini")
#
# with your Gemini model.
#
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-lite-latest"
)


# Gemini embedding model
#
# This replaces:
#
#     OpenAIEmbeddings(
#         model="text-embedding-3-small"
#     )
#
# Gemini's embedding model converts text into
# numerical vectors which FAISS can store/search.
#
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)


# ============================================================
# 2. PDF RETRIEVER STORE
# ============================================================

# Every conversation/thread can have its own
# PDF retriever.
#
# Example:
#
# thread A → resume.pdf
# thread B → notes.pdf
#
# So the PDF knowledge is kept separate
# between conversations.
#
_THREAD_RETRIEVERS: Dict[str, Any] = {}


# Stores metadata about each uploaded document.
#
# Example:
#
# {
#     "abc123": {
#         "filename": "notes.pdf",
#         "documents": 10,
#         "chunks": 35
#     }
# }
#
_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):
    """
    Get the PDF retriever belonging to a thread.

    If the thread doesn't have a PDF,
    return None.
    """

    if (
        thread_id
        and thread_id in _THREAD_RETRIEVERS
    ):
        return _THREAD_RETRIEVERS[thread_id]

    return None


# ============================================================
# 3. PDF INGESTION
# ============================================================

def ingest_pdf(
    file_bytes: bytes,
    thread_id: str,
    filename: Optional[str] = None
) -> dict:
    """
    Process an uploaded PDF and create
    a FAISS vector store for the thread.
    """

    # Make sure a file was actually uploaded
    if not file_bytes:
        raise ValueError(
            "No bytes received for ingestion."
        )


    # --------------------------------------------------------
    # CREATE TEMPORARY PDF FILE
    # --------------------------------------------------------

    # PyPDFLoader expects a file path.
    #
    # Therefore we temporarily save the uploaded
    # PDF bytes to disk.
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:

        temp_file.write(file_bytes)

        temp_path = temp_file.name


    try:

        # ----------------------------------------------------
        # LOAD PDF
        # ----------------------------------------------------

        loader = PyPDFLoader(
            temp_path
        )

        docs = loader.load()


        # ----------------------------------------------------
        # SPLIT PDF INTO SMALL CHUNKS
        # ----------------------------------------------------

        # Large documents are split into smaller
        # pieces so that semantic search works better.
        #
        # Example:
        #
        # PDF
        #  ↓
        # 1000-character chunks
        #  ↓
        # embeddings
        #  ↓
        # FAISS
        #
        splitter = RecursiveCharacterTextSplitter(

            chunk_size=1000,

            chunk_overlap=200,

            separators=[
                "\n\n",
                "\n",
                " ",
                ""
            ]
        )


        chunks = splitter.split_documents(
            docs
        )


        # ----------------------------------------------------
        # CREATE FAISS VECTOR STORE
        # ----------------------------------------------------

        # Gemini embeddings convert every chunk
        # into a numerical vector.
        #
        # FAISS then stores those vectors so that
        # we can perform similarity search later.
        vector_store = FAISS.from_documents(
            chunks,
            embeddings
        )


        # Create a retriever from FAISS
        retriever = vector_store.as_retriever(

            search_type="similarity",

            search_kwargs={
                "k": 4
            }
        )


        # ----------------------------------------------------
        # SAVE RETRIEVER FOR THIS THREAD
        # ----------------------------------------------------

        _THREAD_RETRIEVERS[
            str(thread_id)
        ] = retriever


        # Save document information
        _THREAD_METADATA[
            str(thread_id)
        ] = {

            "filename": (
                filename
                or os.path.basename(temp_path)
            ),

            "documents": len(docs),

            "chunks": len(chunks),
        }


        # Return information that can be shown
        # in the Streamlit UI.
        return {

            "filename": (
                filename
                or os.path.basename(temp_path)
            ),

            "documents": len(docs),

            "chunks": len(chunks),
        }


    finally:

        # ----------------------------------------------------
        # DELETE TEMPORARY PDF
        # ----------------------------------------------------

        # FAISS already has the text/chunks it needs,
        # so the temporary PDF file can be deleted.
        try:

            os.remove(
                temp_path
            )

        except OSError:

            pass


# ============================================================
# 4. TOOLS
# ============================================================


# ------------------------------------------------------------
# DUCKDUCKGO SEARCH
# ------------------------------------------------------------

# Normal LangChain tool.
#
# This is NOT an MCP tool.
#
# It allows Gemini to search the web.
search_tool = DuckDuckGoSearchRun(
    region="us-en"
)


# ------------------------------------------------------------
# CALCULATOR
# ------------------------------------------------------------

@tool
def calculator(
    first_num: float,
    second_num: float,
    operation: str
) -> dict:
    """
    Perform a basic arithmetic operation.

    Supported operations:
        add
        sub
        mul
        div
    """

    try:

        if operation == "add":

            result = (
                first_num
                + second_num
            )


        elif operation == "sub":

            result = (
                first_num
                - second_num
            )


        elif operation == "mul":

            result = (
                first_num
                * second_num
            )


        elif operation == "div":

            if second_num == 0:

                return {
                    "error":
                    "Division by zero is not allowed"
                }

            result = (
                first_num
                / second_num
            )


        else:

            return {
                "error":
                f"Unsupported operation '{operation}'"
            }


        return {

            "first_num": first_num,

            "second_num": second_num,

            "operation": operation,

            "result": result,
        }


    except Exception as e:

        return {
            "error": str(e)
        }


# ------------------------------------------------------------
# RAG TOOL
# ------------------------------------------------------------

@tool
def rag_tool(
    query: str,
    thread_id: Optional[str] = None
) -> dict:
    """
    Retrieve relevant information from the
    uploaded PDF for this chat thread.

    Always provide the thread_id.
    """

    # Get the retriever for this conversation
    retriever = _get_retriever(
        thread_id
    )


    # If no PDF was uploaded
    if retriever is None:

        return {

            "error":
            "No document indexed for this chat. "
            "Upload a PDF first.",

            "query": query,
        }


    # --------------------------------------------------------
    # SEARCH PDF
    # --------------------------------------------------------

    # Search the vector store for chunks
    # semantically related to the user's question.
    result = retriever.invoke(
        query
    )


    # Extract actual text
    context = [
        doc.page_content
        for doc in result
    ]


    # Extract metadata
    metadata = [
        doc.metadata
        for doc in result
    ]


    # Return retrieved information
    return {

        "query": query,

        "context": context,

        "metadata": metadata,

        "source_file": (
            _THREAD_METADATA
            .get(
                str(thread_id),
                {}
            )
            .get(
                "filename"
            )
        ),
    }


# ============================================================
# 5. ALL TOOLS
# ============================================================

# Gemini will be allowed to use these tools:
#
#     DuckDuckGo
#     Calculator
#     PDF RAG
#
tools = [
    search_tool,
    calculator,
    rag_tool
]


# Give all tools to Gemini.
#
# Gemini can now decide whether it needs
# to call a tool.
llm_with_tools = llm.bind_tools(
    tools
)


# ============================================================
# 6. LANGGRAPH STATE
# ============================================================

class ChatState(TypedDict):

    # Stores the conversation messages.
    #
    # add_messages tells LangGraph to append
    # new messages instead of replacing the old ones.
    messages: Annotated[
        list[BaseMessage],
        add_messages
    ]


# ============================================================
# 7. CHAT NODE
# ============================================================

def chat_node(
    state: ChatState,
    config=None
):
    """
    Gemini node.

    Gemini can either:
    
    1. Answer directly
    2. Request a tool
    """

    # --------------------------------------------------------
    # GET THREAD ID
    # --------------------------------------------------------

    thread_id = None


    if (
        config
        and isinstance(config, dict)
    ):

        thread_id = (
            config
            .get("configurable", {})
            .get("thread_id")
        )


    # --------------------------------------------------------
    # SYSTEM MESSAGE
    # --------------------------------------------------------

    # Tell Gemini how it should use the tools.
    system_message = SystemMessage(

        content=(

            "You are a helpful assistant. "

            "For questions about the uploaded PDF, "
            "call the `rag_tool` and include the "
            f"thread_id `{thread_id}`. "

            "Use DuckDuckGo Search when the user "
            "needs current or web information. "

            "Use the calculator tool for arithmetic. "

            "If the user asks about a PDF and no "
            "document is available, ask the user "
            "to upload a PDF."
        )
    )


    # --------------------------------------------------------
    # BUILD MESSAGE LIST
    # --------------------------------------------------------

    # System instruction goes first,
    # followed by the conversation.
    messages = [
        system_message,
        *state["messages"]
    ]


    # --------------------------------------------------------
    # CALL GEMINI
    # --------------------------------------------------------

    response = llm_with_tools.invoke(
        messages,
        config=config
    )


    # Return Gemini's response to LangGraph
    return {
        "messages": [
            response
        ]
    }


# ============================================================
# 8. TOOL NODE
# ============================================================

# ToolNode actually executes the tool requested
# by Gemini.
#
# Example:
#
# Gemini
#   ↓
# "Call calculator"
#   ↓
# ToolNode
#   ↓
# calculator()
#   ↓
# result
#
tool_node = ToolNode(
    tools
)


# ============================================================
# 9. CHECKPOINTER
# ============================================================

# SQLite database stores LangGraph checkpoints.
#
# This allows conversations to be remembered
# using their thread_id.
conn = sqlite3.connect(

    database="chatbot.db",

    check_same_thread=False
)


checkpointer = SqliteSaver(
    conn=conn
)


# ============================================================
# 10. BUILD GRAPH
# ============================================================

graph = StateGraph(
    ChatState
)


# Add Gemini node
graph.add_node(
    "chat_node",
    chat_node
)


# Add tool execution node
graph.add_node(
    "tools",
    tool_node
)


# ------------------------------------------------------------
# GRAPH FLOW
# ------------------------------------------------------------

# START
#   ↓
# Gemini
#
graph.add_edge(
    START,
    "chat_node"
)


# Gemini decides:
#
# Tool needed?
#
# YES → tools
# NO  → END
#
graph.add_conditional_edges(
    "chat_node",
    tools_condition
)


# After the tool executes:
#
# tools
#   ↓
# Gemini
#
# Gemini gets the tool result and produces
# the final answer.
graph.add_edge(
    "tools",
    "chat_node"
)


# ============================================================
# 11. COMPILE CHATBOT
# ============================================================

chatbot = graph.compile(
    checkpointer=checkpointer
)


# ============================================================
# 12. HELPER FUNCTIONS
# ============================================================

def retrieve_all_threads():
    """
    Retrieve all conversation thread IDs
    stored in SQLite.
    """

    all_threads = set()


    # Go through all saved checkpoints
    for checkpoint in checkpointer.list(
        None
    ):

        all_threads.add(
            checkpoint.config[
                "configurable"
            ]["thread_id"]
        )


    return list(
        all_threads
    )


def thread_has_document(
    thread_id: str
) -> bool:
    """
    Check whether a PDF has been uploaded
    for this conversation.
    """

    return (
        str(thread_id)
        in _THREAD_RETRIEVERS
    )


def thread_document_metadata(
    thread_id: str
) -> dict:
    """
    Return metadata about the PDF
    uploaded for this conversation.
    """

    return _THREAD_METADATA.get(
        str(thread_id),
        {}
    )