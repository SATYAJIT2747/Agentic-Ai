import streamlit as st
from langgraph_backend_db import chatbot , llm , retrieve_all_threads
from langchain_core.messages import HumanMessage
import uuid

# **************************************** utility functions *************************
def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def generate_summary(messages):
    if not messages:
        return "New Chat"

    convo = "\n".join(
        f"{m['role']}: {m['content']}" for m in messages
    )

    prompt = f"Give a title in max 8 words.\n\n{convo}"

    response = llm.invoke(prompt)
    return response.content[0]["text"].strip()
def reset_chat():
    current = st.session_state["thread_id"]

    # Save title for current conversation
    if st.session_state["message_history"]:
        title = generate_summary(st.session_state["message_history"])
        st.session_state["chat_titles"][current] = title

    # Create new thread
    new_thread = generate_thread_id()
    st.session_state["thread_id"] = new_thread
    add_thread(new_thread)
    st.session_state["message_history"] = []
def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)
def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    # Check if messages key exists in state values, return empty list if not
    return state.values.get('messages', [])

# **************************************** Session Setup ******************************
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []
if "chat_titles" not in st.session_state:
    st.session_state["chat_titles"] = {}   # {thread_id: summary}
if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()

add_thread(st.session_state['thread_id'])

# **************************************** Sidebar UI *********************************
st.sidebar.title('LangGraph Chatbot')

if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header('My Conversations')
for thread_id in st.session_state['chat_threads'][::-1]:
    title = st.session_state["chat_titles"].get(
        thread_id,
        "New Chat"
    )
    if st.sidebar.button(title, key=str(thread_id)):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []

        for msg in messages:
            if isinstance(msg, HumanMessage):
                role='user'
                content = msg.content
            else:
                role='assistant'
                content = msg.content[0]["text"]
            temp_messages.append({ "role": role,"content": content})

        st.session_state['message_history'] = temp_messages


# **************************************** Main UI ************************************
# loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

user_input = st.chat_input('Type here')
if user_input:
    # User message
    st.session_state["message_history"].append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # AI streaming
    with st.chat_message("assistant"):
        ai_message = st.write_stream(
            msg.content[0]["text"]
            for msg, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config={"configurable": {"thread_id":st.session_state['thread_id']}},
                stream_mode="messages",
            )
            if msg.content and msg.content[0]["text"] #Only stream chunks that contain actual text.
        )

    # Save streamed response
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )
