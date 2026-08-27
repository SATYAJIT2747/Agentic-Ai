# ============================================================
# IMPORTS
# ============================================================

# queue is used to transfer data safely between:
#
#   Background async thread
#            ↓
#       Streamlit UI
#
import queue

# Used to generate unique IDs for conversations
import uuid

# Streamlit is our frontend/UI
import streamlit as st


# Import our LangGraph backend
from langgraph_mcp_backend import (
    chatbot,
    retrieve_all_threads,
    submit_async_task
)


# LangChain message types
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage
)


# ============================================================
# 1. UTILITY FUNCTIONS
# ============================================================

def generate_thread_id():
    """
    Generate a unique ID for every conversation.

    Example:

        Chat 1 → abc123
        Chat 2 → xyz789

    LangGraph uses this ID to keep different
    conversations separate.
    """

    return str(uuid.uuid4())


def reset_chat():
    """
    Start a completely new conversation.
    """

    # Generate a new unique thread ID
    thread_id = generate_thread_id()

    # Make this the current conversation
    st.session_state["thread_id"] = thread_id

    # Add the new conversation to our list
    add_thread(thread_id)

    # Clear messages currently displayed
    st.session_state["message_history"] = []


def add_thread(thread_id):
    """
    Add a conversation ID to the list
    if it doesn't already exist.
    """

    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(
            thread_id
        )


def load_conversation(thread_id):
    """
    Load a previously saved conversation
    from LangGraph's checkpointer.
    """

    # Ask LangGraph for the saved state
    # of this particular thread.
    state = chatbot.get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    # Get the messages from the state.
    #
    # If there are no messages, return [].
    return state.values.get(
        "messages",
        []
    )


# ============================================================
# 2. STREAMLIT SESSION STATE
# ============================================================

# Streamlit reruns this Python file whenever
# the user interacts with the UI.
#
# st.session_state allows us to preserve
# information between those reruns.


# Store messages shown in the current chat
if "message_history" not in st.session_state:

    st.session_state["message_history"] = []


# Create a thread ID for the current conversation
if "thread_id" not in st.session_state:

    st.session_state["thread_id"] = generate_thread_id()


# Retrieve previously saved conversation IDs
# from the LangGraph SQLite checkpointer.
if "chat_threads" not in st.session_state:

    st.session_state["chat_threads"] = (
        retrieve_all_threads()
    )


# Make sure the current thread exists
# in the conversation list.
add_thread(
    st.session_state["thread_id"]
)


# ============================================================
# 3. SIDEBAR
# ============================================================

st.sidebar.title(
    "LangGraph MCP Chatbot"
)


# ------------------------------------------------------------
# NEW CHAT BUTTON
# ------------------------------------------------------------

if st.sidebar.button("New Chat"):

    reset_chat()


# ------------------------------------------------------------
# PREVIOUS CONVERSATIONS
# ------------------------------------------------------------

st.sidebar.header(
    "My Conversations"
)


# Reverse the list so the latest conversations
# appear at the top.
for thread_id in st.session_state["chat_threads"][::-1]:

    # Create one button for every conversation
    if st.sidebar.button(str(thread_id)):

        # Make the selected thread the current thread
        st.session_state["thread_id"] = thread_id


        # Load messages belonging to this thread
        messages = load_conversation(
            thread_id
        )


        # Convert LangChain messages into
        # a simple format for Streamlit.
        temp_messages = []


        for msg in messages:

            # HumanMessage → user
            #
            # Other messages → assistant
            role = (
                "user"
                if isinstance(msg, HumanMessage)
                else "assistant"
            )


            temp_messages.append({

                "role": role,

                "content": msg.content
            })


        # Replace the current UI history
        # with the selected conversation.
        st.session_state["message_history"] = (
            temp_messages
        )


# ============================================================
# 4. MAIN CHAT UI
# ============================================================


# ------------------------------------------------------------
# DISPLAY PREVIOUS MESSAGES
# ------------------------------------------------------------

for message in st.session_state["message_history"]:

    with st.chat_message(
        message["role"]
    ):

        st.text(
            message["content"]
        )


# ------------------------------------------------------------
# USER INPUT
# ------------------------------------------------------------

user_input = st.chat_input(
    "Type here"
)


# ============================================================
# 5. WHEN USER SENDS A MESSAGE
# ============================================================

if user_input:

    # --------------------------------------------------------
    # SHOW USER MESSAGE IMMEDIATELY
    # --------------------------------------------------------

    # Save the user's message in Streamlit history
    st.session_state["message_history"].append({

        "role": "user",

        "content": user_input
    })


    # Display user's message
    with st.chat_message("user"):

        st.text(user_input)


    # ========================================================
    # LANGGRAPH CONFIGURATION
    # ========================================================

    # thread_id tells LangGraph:
    #
    # "Which conversation does this message
    #  belong to?"
    #
    # metadata and run_name are useful for
    # tracing/debugging.
    CONFIG = {

        "configurable": {

            "thread_id": (
                st.session_state["thread_id"]
            )
        },

        "metadata": {

            "thread_id": (
                st.session_state["thread_id"]
            )
        },

        "run_name": "chat_turn",
    }


    # ========================================================
    # 6. ASSISTANT MESSAGE AREA
    # ========================================================

    with st.chat_message("assistant"):

        # ----------------------------------------------------
        # TOOL STATUS HOLDER
        # ----------------------------------------------------

        # We will store the Streamlit status box here.
        #
        # Initially:
        #
        #     box = None
        #
        # When a tool runs:
        #
        #     box = st.status(...)
        #
        status_holder = {
            "box": None
        }


        # ====================================================
        # 7. STREAM LANGGRAPH RESPONSE
        # ====================================================

        def ai_only_stream():
            """
            Stream LangGraph events and yield
            only the actual AI text to Streamlit.
            """

            # ------------------------------------------------
            # QUEUE
            # ------------------------------------------------

            # This queue is the bridge between:
            #
            #   Async background task
            #           ↓
            #       event_queue
            #           ↓
            #       Streamlit
            #
            # The async function puts data into
            # this queue.
            #
            # The normal Python code reads it.
            event_queue: queue.Queue = (
                queue.Queue()
            )


            # =================================================
            # ASYNC STREAM FUNCTION
            # =================================================

            async def run_stream():
                """
                Run LangGraph asynchronously
                and put every streamed event
                into the queue.
                """

                try:

                    # chatbot.astream() streams the
                    # LangGraph execution as it happens.
                    #
                    # async for means:
                    #
                    # "Keep receiving streamed events
                    #  asynchronously."
                    async for message_chunk, metadata in (
                        chatbot.astream(

                            {
                                "messages": [
                                    HumanMessage(
                                        content=user_input
                                    )
                                ]
                            },

                            config=CONFIG,

                            # We want message-level
                            # streaming events.
                            stream_mode="messages",
                        )
                    ):

                        # Put the chunk into the queue
                        # so Streamlit can process it.
                        event_queue.put(
                            (
                                message_chunk,
                                metadata
                            )
                        )


                except Exception as exc:

                    # If something goes wrong,
                    # send the error through the queue.
                    event_queue.put(
                        (
                            "error",
                            exc
                        )
                    )


                finally:

                    # None is our signal that
                    # streaming is finished.
                    event_queue.put(
                        None
                    )


            # =================================================
            # START ASYNC TASK
            # =================================================

            # submit_async_task() sends run_stream()
            # to the backend's dedicated asyncio
            # event loop.
            #
            # We don't directly call:
            #
            #     asyncio.run()
            #
            # because our backend already has
            # an async event loop running.
            submit_async_task(
                run_stream()
            )


            # =================================================
            # READ EVENTS FROM QUEUE
            # =================================================

            while True:

                # Wait until the async function
                # puts something into the queue.
                item = event_queue.get()


                # None means streaming has finished
                if item is None:

                    break


                # Separate the message and metadata
                message_chunk, metadata = item


                # ------------------------------------------------
                # HANDLE ERRORS
                # ------------------------------------------------

                if message_chunk == "error":

                    raise metadata


                # =================================================
                # 8. TOOL MESSAGE
                # =================================================

                # ToolMessage means a tool was executed.
                #
                # This could be:
                #
                #     DuckDuckGo
                #
                # OR:
                #
                #     Calculator MCP
                #
                if isinstance(
                    message_chunk,
                    ToolMessage
                ):

                    # Get the tool name
                    tool_name = getattr(
                        message_chunk,
                        "name",
                        "tool"
                    )


                    # If this is the first tool event,
                    # create the status box.
                    if status_holder["box"] is None:

                        status_holder["box"] = (
                            st.status(
                                f"🔧 Using `{tool_name}` …",
                                expanded=True
                            )
                        )


                    # If the status box already exists,
                    # update it.
                    else:

                        status_holder["box"].update(

                            label=(
                                f"🔧 Using `{tool_name}` …"
                            ),

                            state="running",

                            expanded=True
                        )


                # =================================================
                # 9. AI MESSAGE
                # =================================================

                # Gemini's response comes through
                # as AIMessage chunks.
                if isinstance(
                    message_chunk,
                    AIMessage
                ):

                    # Get Gemini's content
                    content = message_chunk.content


                    # ------------------------------------------------
                    # GEMINI MAY RETURN A LIST
                    # ------------------------------------------------

                    # Sometimes Gemini returns:
                    #
                    # [
                    #     {
                    #         "type": "text",
                    #         "text": "Hello"
                    #     }
                    # ]
                    #
                    # We only want the "text".
                    if isinstance(
                        content,
                        list
                    ):

                        for block in content:

                            # If this is a dictionary,
                            # extract its "text" field.
                            if isinstance(
                                block,
                                dict
                            ):

                                text = block.get(
                                    "text",
                                    ""
                                )

                                if text:
                                    yield text


                            # Sometimes the block itself
                            # is a string.
                            elif isinstance(
                                block,
                                str
                            ):

                                if block:
                                    yield block


                    # ------------------------------------------------
                    # GEMINI MAY RETURN A STRING
                    # ------------------------------------------------

                    elif isinstance(
                        content,
                        str
                    ):

                        if content:
                            yield content


        # ========================================================
        # 10. WRITE STREAM TO STREAMLIT
        # ========================================================

        # st.write_stream() receives the text
        # produced by ai_only_stream().
        #
        # Instead of waiting for the entire answer:
        #
        #     "The answer is 100"
        #
        # it can display:
        #
        #     "The"
        #     " answer"
        #     " is"
        #     " 100"
        #
        # as the response arrives.
        ai_message = st.write_stream(
            ai_only_stream()
        )


        # ========================================================
        # 11. FINISH TOOL STATUS
        # ========================================================

        # If a tool was used, close the status box.
        if status_holder["box"] is not None:

            status_holder["box"].update(

                label="✅ Tool finished",

                state="complete",

                expanded=False
            )


    # ============================================================
    # 12. SAVE ASSISTANT MESSAGE
    # ============================================================

    # Save the final assistant response
    # in Streamlit's session state.
    #
    # This is needed because Streamlit reruns
    # the script after interactions.
    st.session_state["message_history"].append({

        "role": "assistant",

        "content": ai_message
    })