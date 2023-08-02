
import streamlit as st
import os
from streamlit_chat import message
from PIL import Image
from pathlib import Path
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import StdOutCallbackHandler
from ingest import AllResourcesFactory
from typing import List


#Factory that returns Keys, Templates, Vectorstore and LLms class instances
resource_factory = AllResourcesFactory()

def get_qa_chain() -> ConversationalRetrievalChain:
    llm = resource_factory.llms
    retriever = resource_factory.vectorstore.as_retriever(llm)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    memory.save_context({'input': resource_factory.templates.qa_template},
                        {'output': 'Yes, I will only answer questions related to the context in the provided document'})
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm.get_llm(),
        retriever,
        condense_question_prompt=resource_factory.templates.condense_question_prompt, 
        memory=memory,
        verbose=True,
        callbacks=[StdOutCallbackHandler()]
    )
    return qa_chain

def main():
    #images
    ai_thumbnail = Path.cwd().joinpath("images","Ai-Chatbot.jpg.jpg")
    css_file = Path.cwd().joinpath("styles","main.css")

    #Scalers
    PAGE_TITLE = "Contrail Knowledge Bot"
    PAGE_ICON = "💬"

    #st config
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON)

    #Load css style content
    with open(css_file) as css:
        st.markdown("<style>{}</style>".format(css.read()), unsafe_allow_html=True)
    ai_thumbnail = Image.open(ai_thumbnail)

    #define st page
    col1, col2 = st.columns(2)
    with col1:
        st.title("Juniper Contrail Knowledge Base Chatbot")
        uploaded_file = st.file_uploader("Upload Your own Documents in PDF format")
        if uploaded_file is not None and uploaded_file.name not in os.listdir("data"):
            with open("data/" + uploaded_file.name, "wb") as uf:
                uf.write(uploaded_file.getbuffer())
            st.write("File uploaded successfully")
            with st.spinner("Vectorizing the data store"):
                resource_factory.vectorstore.load_and_split_data()  
                st.session_state.qa_chain = get_qa_chain()
                st.success("Done!")
    with col2:
        st.image(ai_thumbnail, width=230)

    if not os.path.isfile(os.getcwd()+'/vectorstore.pkl'):
        st.write("No Files Found, Please upload a document")
        st.stop()

    if 'generated' not in st.session_state:
        st.session_state["generated"] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []

    if 'qa_chain' not in st.session_state:
        with st.spinner("Getting Ready..."):
            st.session_state.qa_chain = get_qa_chain()

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history:List[(tuple)] = []

    col1, col2 = st.columns(2, gap='small')
    with col1:
        chatbox = st.empty()
        user_input = chatbox.text_input("You: ", value='')
        if user_input:
            with st.spinner("Thinking..."):
                output = st.session_state.qa_chain.run(
                    question = user_input,
                    #QA_PROMPT=resource_factory.templates.qa_prompt, 
                )
            print (output)
            st.session_state.past.append(user_input)
            #print(st.session_state.past)
            st.session_state.generated.append(output)
            #print(st.session_state.generated)
            st.session_state.chat_history.append((user_input, output))

    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"])-1, -1, -1):
            message(st.session_state.generated[i], key=str(i))
            message(st.session_state.past[i], is_user=True, key=str(i)+"_user")

if __name__ == '__main__':
    main()