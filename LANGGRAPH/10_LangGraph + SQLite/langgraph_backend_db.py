from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage , HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langsmith import traceable
load_dotenv()
import os
os.environ['LANGCHAIN_PROJECT'] = 'chat-bot-dashboard_v2'
llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest")

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
@traceable
def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}
conn = sqlite3.connect(database="chatbot_db" , check_same_thread= False)
# Checkpointer
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)
# config = {'configurable':{'thread_id':'thread1'}}
# response = chatbot.invoke(
#     {
#         "messages": [HumanMessage(content="what is  my name ")]
#     },
#     config=config
# )
# print(response)
def retrieve_all_threads():
    all_threads = set()

    # None = retrieve checkpoints from all threads
    for checkpoint in checkpointer.list(None):
        thread_id = checkpoint.config["configurable"]["thread_id"]
        all_threads.add(thread_id)

    return list(all_threads)