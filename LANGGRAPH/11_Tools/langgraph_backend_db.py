from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage , HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langsmith import traceable
from langgraph.prebuilt import ToolNode , tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
load_dotenv()
import os
os.environ['LANGCHAIN_PROJECT'] = 'chat-bot-Tools'
llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest")
# -------------------
# 2. Tools
# -------------------


search_tool = DuckDuckGoSearchRun(region="us-en")
@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}
tools = [search_tool , calculator]
llm_with_tools = llm.bind_tools(tools)











class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
@traceable
def chat_node(state: ChatState):
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    print("TOOL CALLS:", response.tool_calls)
    return {"messages": [response]}
tool_node = ToolNode(tools)
conn = sqlite3.connect(database="chatbot_db" , check_same_thread= False)
# Checkpointer
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node('tools' , tool_node)
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition) # we have already bind our llm with tooll na so when ai interpret our msg it knows it have to call a tool or not ok.
graph.add_edge('tools' , 'chat_node')

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