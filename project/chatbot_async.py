from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage , ToolMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import ToolNode, tools_condition
import sqlite3
import _asyncio




load_dotenv()

search = DuckDuckGoSearchRun(name="duckduckgo_search")

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """Performs a basic arithmetic operation on two numbers. Supports add, sub, mul, div."""
    try:
        print("ansh ")
        if operation == "add": return {"result": first_num + second_num}
        elif operation == "sub": return {"result": first_num - second_num}
        elif operation == "mul": return {"result": first_num * second_num}
        elif operation == "div": return {"result": first_num / second_num}
        else: return {"error": f"Unsupported operation: {operation}"}
    except Exception as e:
        return {"error": str(e)}

tools = [search, calculator]

model1 = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)

model =model1.bind_tools(tools)




class ChatBot(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def build_graph():
    def chat_node(state: ChatBot) -> ChatBot:
        messages = state['messages']

        model_response = model1.ainvoke(messages)

        return {'messages': [model_response]}


    graph = StateGraph(ChatBot)

    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", ToolNode(tools))

    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")

    graph.compile()

    connection = sqlite3.connect(database ='chatbot.db', check_same_thread=False)
    checkpointer = SqliteSaver(connection)

    workflow = graph.compile(checkpointer=checkpointer)
   
    return chatbot

async def main():
    chatbot = build_graph()

    response = model1.ainvoke(
             {"messages": [HumanMessage(content='what is india capital')]})

    print(response.text)

if __name__ == '__main__':
    _asyncio.run(main())




def chat_node(state: ChatBot) -> ChatBot:
    messages = state['messages']

    model_response = model.invoke(messages)

    return {'messages': [model_response]}


graph = StateGraph(ChatBot)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

graph.compile()

connection = sqlite3.connect(database ='chatbot.db', check_same_thread=False)
checkpointer = SqliteSaver(connection)

workflow = graph.compile(checkpointer=checkpointer)
