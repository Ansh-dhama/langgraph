 
import sqlite3
from pathlib import Path
from typing import Annotated, Optional, TypedDict

from dotenv import load_dotenv

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_huggingface import HuggingFaceEmbeddings ,ChatHuggingFace ,HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from pypdf import PdfReader
from io import BytesIO
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableConfig



load_dotenv()

search = DuckDuckGoSearchRun(name="duckduckgo_search",
                              description=(
        "Search the web ONLY when the user needs current information, "
        "recent news, live information, or facts that you cannot answer "
        "reliably from your existing knowledge. "
        "Do NOT use for greetings, casual conversation, basic explanations, "
        "programming concepts, or simple calculations."
    ),)

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
embeddings = HuggingFaceEmbeddings(
    model_name="google/embeddinggemma-300m",
    encode_kwargs={
        "normalize_embeddings": True,
        "truncate_dim": 128,
    },
    query_encode_kwargs={
        "normalize_embeddings": True,
        "truncate_dim": 128,
    },
)

vector_stores = {}

def ingest_pdf(
    file_bytes: bytes,
    thread_id: str,
    filename: Optional[str] = None,
) -> dict:
    """Extract PDF text, split it into chunks, create embeddings,
    and store the vectors in a FAISS index for this conversation."""

    try:
        # 1. Read the uploaded PDF
        reader = PdfReader(BytesIO(file_bytes))

        if reader.is_encrypted:
            return {
                "success": False,
                "error": "Encrypted PDFs are not supported.",
            }

        # 2. Extract text from every page
        documents = []

        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""

            if page_text.strip():
                documents.append(
                    {
                        "text": page_text,
                        "page": page_number,
                    }
                )

        if not documents:
            return {
                "success": False,
                "error": (
                    "No extractable text found. "
                    "The PDF may be scanned and require OCR."
                ),
            }

        # 3. Split extracted text into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

        chunks = []

        for page in documents:
            page_chunks = splitter.create_documents(
                [page["text"]],
                metadatas=[
                    {
                        "thread_id": thread_id,
                        "filename": filename or "document.pdf",
                        "page": page["page"],
                    }
                ],
            )
            chunks.extend(page_chunks)

        if not chunks:
            return {
                "success": False,
                "error": "No text chunks were generated.",
            }

        # 4. Generate embeddings and create the FAISS index
        new_vector_store = FAISS.from_documents(
            chunks,
            embeddings,
        )
        print("Number of PDF chunks:", len(chunks))
        print("Embedding dimensions:", new_vector_store.index.d)
        print("FAISS vectors stored:", new_vector_store.index.ntotal)
        print("First chunk:", chunks[0].page_content[:200])
        # 5. Store or merge the index for this thread
        if thread_id in vector_stores:
            vector_stores[thread_id].merge_from(new_vector_store)
        else:
            vector_stores[thread_id] = new_vector_store
                
       

        return {
            "success": True,
            "filename": filename or "document.pdf",
            "pages_extracted": len(documents),
            "chunks_created": len(chunks),
            "message": "PDF embedded and indexed successfully.",
        }

    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
        }


@tool
def rag_tool(query: str, config: RunnableConfig) -> dict:
    """Search the PDF uploaded in the current conversation.
    Use ONLY when the user explicitly asks about an uploaded PDF
    or asks a question that requires information from that PDF."""

    thread_id = config["configurable"]["thread_id"]

    store = vector_stores.get(thread_id)

    if store is None:
        return {
            "error": (
                f"No PDF is indexed for conversation {thread_id}. "
                "Ask the user to upload and process a PDF first."
            )
        }

    docs = store.similarity_search(query, k=4)

    return {
        "context": [
            doc.page_content for doc in docs
        ],
        "metadata": [
            doc.metadata for doc in docs
        ],
    }

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)

tools = [search, calculator, rag_tool]

model = llm.bind_tools(tools)


class ChatBot(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


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


connection = sqlite3.connect(database ='chatbot.db', check_same_thread=False)
checkpointer = SqliteSaver(connection)

workflow = graph.compile(checkpointer=checkpointer)


def retrive_all_thread():
   all_thread = set()
   for checkpoint in checkpointer.list(None):
     all_thread.add(checkpoint.config['configurable']['thread_id'])
   return list(all_thread)


# for message_chunk, metadata in workflow.stream(
#     {'messages': [HumanMessage(content='Written a essay on india capital in 100 words')]},
#     config=CONFIG,
#     stream_mode='messages'
# ):
#   print(message_chunk.content, end='', flush=True)

# print()

# Runs only when this backend file is executed directly.

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

#         messages = response["messages"]
#         latest_user_index = max(
#             i for i, msg in enumerate(messages)
#             if isinstance(msg, HumanMessage)
#         )

#         current_turn = messages[latest_user_index:]
#         tool_executed = False

#         for msg in current_turn:

#             for call in getattr(msg, "tool_calls", []):
#                 print(f"\nLLM requested tool: {call['name']}")
#                 print("Arguments:", call["args"])

#             if isinstance(msg, ToolMessage):
#                 tool_executed = True
#                 print(f"\nTool executed: {msg.name}")
#                 print("Tool result:", msg.content)

#         if not tool_executed:
#             print("\nNo tool was executed in this turn.")

#         print("\nBot:", messages[-1].content)
