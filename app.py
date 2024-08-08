import os
import csv
import docx
import io
import streamlit as st
from streamlit_chat import message
from langchain_openai import OpenAI
from pypdf import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.callbacks import get_openai_callback
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline






openai_api_key = st.secrets['OPENAI_API_KEY']




def main():
    st.set_page_config("Welcome to document.ai")
    st.title('Document.ai')

    if 'process_completed' not in st.session_state:
        st.session_state.process_completed = None
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None


    
    with st.sidebar:
       upload_documents = st.file_uploader(label='Upload your file', type=['pdf', 'csv', 'docx'], accept_multiple_files=True)
       process = st.button(label='Process')

    if(process):
        print('Process pressed')
        text = get_text_from_file(upload_documents)
        st.write('Get text from file')

        text_chunks = get_text_split(text)
        st.write('chunke created')

        vectorstores = get_vectorstores(text_chunks)
        st.write('vecter created')
        st.session_state.conversation = get_conversation_chain(vectorstores, openai_api_key) #for openAI

        st.session_state.process_completed = True

    if  (st.session_state.process_completed == True):
        user_input = st.chat_input("Ask question aout your file")
        if(user_input):
            handle_user_input(user_input) 


          
          

          



def get_text_split(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=900,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

        

def get_text_from_file(upload_documents):
    text = ''
    for file in upload_documents:
        split_file_name = os.path.splitext(file.name)
        file_extension = split_file_name[1]
        if (file_extension == ".pdf"):
            text+= get_read_pdf(file)
            return text
        elif (file_extension == '.docx'):
            text+= get_read_docx(file)
            


def get_read_pdf(file):
    loader = PdfReader(file)
    text = ''
    pages = loader.pages
    for page in pages:
        text+= page.extract_text()
    return text


def get_read_docx(file):
    text = ''
    doc = docx.Document(file)
    for i in doc.paragraphs:
        text+= i.text
    return text



def get_read_csv(file):
    file.seek(0)
    with io.StringIO(file.getvalue().decode("utf-8")) as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            print(row)


def get_vectorstores(text_chunks):
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    db = FAISS.from_texts(text_chunks, embedding)
    return db

# with openai llm

def get_conversation_chain(vetorestore,openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name = 'gpt-3.5-turbo',temperature=0)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vetorestore.as_retriever(),
        memory=memory
    )
    
    return conversation_chain


#with free hugingface llm

# def get_conversation_chain(vetorestore):
#     model_id = "gp2"
#     tokenizer = AutoTokenizer.from_pretrained(model_id)
#     model = AutoModelForCausalLM.from_pretrained(model_id)
#     pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10)
#     llm = HuggingFacePipeline(pipeline=pipe)
#     memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
#     conversation_chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=vetorestore.as_retriever(),
#         memory=memory
#     )
#     return conversation_chain


def handle_user_input(user_input):
    with get_openai_callback() as cb:
        response =  st.session_state.conversation({'question':user_input})
        st.session_state.chat_history = response['chat_history']

    response_container = st.container()

    with response_container:
        for i, messages in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                message(messages.content, is_user=True, key=str(i))
            else:
                message(messages.content, key=str(i))




if __name__ == '__main__':
    main()
