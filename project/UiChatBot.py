import streamlit as st
from langgraph_backend import workflow
from langchain_core.messages import HumanMessage
import uuid

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
    st.session_state['chat_thread'] = []

add_thread(st.session_state['thread_id'])



st.sidebar.title('Langgraph chatbot')
if st.sidebar.button('New button'):
    reset_chat()




st.sidebar.header('My conversation')

for thread_id in st.session_state['chat_thread'][::-1]:

   if st.sidebar.button(str(thread_id) , key=f'chat_{thread_id}'):
       st.session_state['thread_id'] = thread_id
       mess = load_conversation(thread_id)

       temp_messeges = []

       for m in mess:
           if isinstance(m , HumanMessage):
               role = 'user'
           else:
               role = 'assistant'
           temp_messeges.append({'role': role , 'content': m.text})
       st.session_state['messages'] = temp_messeges


CONFIG = {
    'configurable': {
        'thread_id':st.session_state['thread_id']
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
        st.text(user_input)
    with st.chat_message('assistant'):
        Ai_messege =  st.write_stream(
            messege_chunk.text for messege_chunk , metadata in workflow.stream(
                  {'messages' :[HumanMessage(content=user_input)]},
                     config=CONFIG,
                     stream_mode='messages'
             )
         )
    st.session_state['messages'].append({
            'role': 'assistant',
            'content': Ai_messege
        })