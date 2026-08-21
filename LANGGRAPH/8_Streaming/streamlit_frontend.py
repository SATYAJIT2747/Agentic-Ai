import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage

CONFIG = {"configurable": {"thread_id": "thread-1"}}

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

# Show old messages
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Type here")

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
                config=CONFIG,
                stream_mode="messages",
            )
            if msg.content and msg.content[0]["text"] #Only stream chunks that contain actual text.
        )

    # Save streamed response
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )