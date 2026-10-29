import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid
from langgraph_backend import workflow, retrive_all_thread, ingest_pdf

def  getUtility():
    thread_id  = uuid.uuid4()
    return str(thread_id)

def reset_chat():
    thread_id = getUtility()
    st.session_state['thread_id']= thread_id
    add_thread(thread_id)
    st.session_state['messages'] = []

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_thread']:
        st.session_state['chat_thread'].append(thread_id)

def load_conversation(thread_id):
        config = {
        'configurable': {
            'thread_id': thread_id
        }
    }
        return workflow.get_state(config=config).values.get('messages',[])

if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = getUtility()
if 'chat_thread' not in st.session_state:
    st.session_state['chat_thread'] = retrive_all_thread()

add_thread(st.session_state['thread_id'])



st.sidebar.title('Langgraph chatbot')
if st.sidebar.button('New button'):
    reset_chat()




st.sidebar.header('My conversation')


st.sidebar.header("Upload PDF")

uploaded_file = st.sidebar.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)

if uploaded_file is not None:
    if st.sidebar.button("Process PDF"):
        with st.spinner("Extracting text and creating embeddings..."):
            result = ingest_pdf(
                file_bytes=uploaded_file.getvalue(),
                thread_id=st.session_state["thread_id"],
                filename=uploaded_file.name,
            )

        if result["success"]:
            st.sidebar.success(
                f"PDF processed! Chunks: {result['chunks_created']}"
            )
        else:
            st.sidebar.error(result["error"])


for thread_id in st.session_state["chat_thread"][::-1]:
    if st.sidebar.button(str(thread_id), key=f"chat_{thread_id}"):
        st.session_state["thread_id"] = thread_id
        mess = load_conversation(thread_id)

        temp_messages = []

        for m in mess:
            if isinstance(m, HumanMessage):
                role = "user"
            elif isinstance(m, AIMessage):
                role = "assistant"
            else:
                continue

            temp_messages.append({
                "role": role,
                "content": m.content
            })

        st.session_state["messages"] = temp_messages
        st.rerun()

CONFIG = {
    'configurable': {
        'thread_id':st.session_state['thread_id']
    },
    'metadata':{
        "thread_id":st.session_state["thread_id"]
    }
 
}

for m in st.session_state['messages']:
    with st.chat_message(m['role']):
        st.text(m['content'])



user_input = st.chat_input("Type your message here...")

if user_input:
    st.session_state['messages'].append({
        'role': 'user',
        'content': user_input
    })

    with st.chat_message('user'):
        st.write(user_input)

    final_response = ""

    with st.chat_message('assistant'):

        with st.status("Agent processing...", expanded=True) as status:

            try:
                for event in workflow.stream(
                    {'messages': [HumanMessage(content=user_input)]},
                    config=CONFIG,
                    stream_mode="updates"
                ):

                    for node_name, node_update in event.items():

                        node_messages = node_update.get("messages", [])

                        # 1. LLM node: identify the selected tool
                        if node_name == "chat_node":

                            for message in node_messages:

                                if not isinstance(message, AIMessage):
                                    continue

                                if message.tool_calls:
                                    for call in message.tool_calls:
                                        status.write(
                                            f"🧠 LLM selected tool: "
                                            f"`{call['name']}`"
                                        )

                                        status.write(
                                            f"Arguments: {call['args']}"
                                        )

                                else:
                                    # Final LLM response
                                    content = message.text

                                    if isinstance(content, str):
                                        final_response = content
                                    else:
                                        final_response = str(content)

                        # 2. Tools node: show execution and result
                        elif node_name == "tools":

                            status.write(
                                "🛠️ Tool Executor: Executing tool..."
                            )

                            for message in node_messages:

                                if isinstance(message, ToolMessage):
                                    status.write(
                                        f"✅ Executed: `{message.name}`"
                                    )

                                    status.write(
                                        f"Tool result: {message.content}"
                                    )

                status.update(
                    label="Execution completed!",
                    state="complete",
                    expanded=False
                )

            except Exception as e:
                status.update(
                    label="Execution failed!",
                    state="error",
                    expanded=True
                )
                st.error(f"Error: {e}")

        # 3. Display the final answer in the chat
        if final_response:
            st.markdown(final_response)

            st.session_state['messages'].append({
                'role': 'assistant',
                'content': final_response
            })
