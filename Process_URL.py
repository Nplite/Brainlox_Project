import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlT import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from transformers import GPT2TokenizerFast
import altair as alt
from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFLoader
import PyPDF2
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())
from langchain.document_loaders import UnstructuredURLLoader
import requests
from bs4 import BeautifulSoup




# Set Slack API credentials
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# url = ["https://brainlox.com/courses/category/technical"]
# loaders = UnstructuredURLLoader(urls=url)
# data = loaders.load()



def load_text_from_url(url):
  res = requests.get(url)
  html = res.text

  soup = BeautifulSoup(html, 'html.parser')

  # Remove script and style tags
  for script in soup(["script", "style"]):
    script.extract()

  # Get text
  text = soup.get_text()

  # Break into lines and remove leading and trailing space on each
  lines = (line.strip() for line in text.splitlines())
  # Break multi-headlines into a line each
  chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
  # Drop blank lines
  text = '\n'.join(chunk for chunk in chunks if chunk)

  return text

url = "https://brainlox.com/courses/category/technical"
data = load_text_from_url(url)

# # You MUST add your PDF to local files in this notebook (folder icon on left hand side of screen)
# def extract_text_from_pdf(file_path):
#     with open(file_path, 'rb') as file:
#         pdf_reader = PyPDF2.PdfReader(file)
#         text = ""
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text

# pdf_text = extract_text_from_pdf('./Documents/export_to_europe_guide.pdf')

# Create the "Text_Data" folder if it doesn't exist
folder_path = 'Text_Data'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Step 2: Save to .txt and reopen (helps prevent issues)
file_path = os.path.join('Text_Data', 'Brainlox_data.txt')
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(data)

with open('./Text_Data/Brainlox_data.txt', 'r', encoding='utf-8') as f:
    text = f.read()



def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore 



def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(openai_api_key = OPENAI_API_KEY)
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():

    st.write(css, unsafe_allow_html=True)

    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    styl = f"""
    <style>
        .stTextInput {{
        position: fixed;
        bottom: 3rem;
    }}
    </style>
    """

    st.markdown(styl, unsafe_allow_html=True)



    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    
    user_question = st.text_input("About Trade Compliance:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.header("Trade Chat :books:")

        with st.spinner("Processing"):
                # get pdf text
                
            raw_text = os.path.join(text) 

                # get the text chunks
            text_chunks = get_text_chunks(raw_text)

                # create vector store
            vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
            st.session_state.conversation = get_conversation_chain(
                vectorstore)
               
        st.markdown("[Made by: Varahi Technology](https://varahitechnologies.com/)")




if __name__ == '__main__':
    main()

