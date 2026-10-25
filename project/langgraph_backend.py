from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)

class ChatBot(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatBot) -> ChatBot:
    messages = state['messages']

    model_response = model.invoke(messages)

    return {'messages': [model_response]}


graph = StateGraph(ChatBot)

graph.add_node('chat_node', chat_node)
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

checkpointer = InMemorySaver()
workflow = graph.compile(checkpointer=checkpointer)

# for message_chunk, metadata in workflow.stream(
#     {'messages': [HumanMessage(content='Written a essay on india capital in 100 words')]},
#     config=CONFIG,
#     stream_mode='messages'
# ):
#   print(message_chunk.content, end='', flush=True)

# print()

# # Runs only when this backend file is executed directly.
# if __name__ == "__main__":
#     CONFIG = {
#         "configurable": {
#             "thread_id": "thread-1"
#         }
#     }

#     while True:
#         user_input = input("You: ")

#         if user_input.strip().lower() in ("exit", "quit"):
#             break

#         if not user_input.strip():
#             continue

#         response = workflow.invoke(
#             {"messages": [HumanMessage(content=user_input)]},
#             config=CONFIG
#         )

#         print("Bot:", response["messages"][-1].text)