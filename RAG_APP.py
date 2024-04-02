import os
import streamlit as st
import pinecone
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Pinecone
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import OpenAIEmbeddings
from pinecone import PodSpec

# Set environment variables

OPENAI_API =st.secrets['OPENAI_API_KEY']
api = st.secrets['api_key'] 
PINECONE_API_KEY = st.secrets['PINECONE_API_KEY'] 

os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
os.environ['OPENAI_API_KEY'] = OPENAI_API
os.environ["api_key"] = api

llm = ChatOpenAI()

# Function to load documents
def load_documents(file):
    name, extension = os.path.splitext(file)
    if extension == ".pdf":
        loader = PyPDFLoader(file)
    elif extension == ".docx":
        loader = Docx2txtLoader(file)
    elif extension == ".txt":
        loader = TextLoader(file, encoding='iso-8859-1')
    else:
        st.error("Document format is not supported!")
        return None
    return loader.load()

# Function to load web sources
def load_external(source):
    if source:
        loader = WebBaseLoader(source)
        return loader.load()
    else:
        st.error("Web address does not support web scraping!")
        return None

# Function to chunk data
def chunk_data(data, chunk_size=1000):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=100)
    return text_splitter.split_documents(data)

# Function to create embeddings vector store
def create_embeddings_vectorstore(chunked_data):
    # importing the necessary libraries and initializing the Pinecone client
    
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)

    import os
    from pinecone import Pinecone, PodSpec
    pc = Pinecone(
        api_key=os.environ.get("PINECONE_API_KEY")
    )
    index_name = "project"
    if index_name in pc.list_indexes().names():
        vector_store = pinecone.from_existing_index(index_name, embeddings)
    else:
        # creating a new index
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric='cosine',
            spec=PodSpec(
                environment='gcp-starter'
            )
        )
        vector_store = pc.from_documents(chunked_data, embeddings, index_name=index_name)
    return vector_store

# Function to delete Pinecone index
def delete_pinecone_index(index_name='project'):
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment='gcp-starter')
    pc = pinecone.Pinecone()
    
    if index_name == 'all':
        indexes = pc.list_indexes().names()
        
        for index in indexes:
            pc.delete_index(index)
        
    else:
        pc.delete_index(index_name)

# Function to ask and get answers
def questions_answer(question, vector_store):
    retriever_from_llm = MultiQueryRetriever.from_llm(vector_store.as_retriever(search_type='similarity'), llm=llm)
    template = """
        Answer the question based only on the following context:
        {context} and reply If you can't answer then return `I DONT KNOW`.

        Question: {question}
        """
    prompt = ChatPromptTemplate.from_template(template)
    chain = (
        {"context": retriever_from_llm, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain.invoke(question)

def clear_vector_store():
    
    if 'vs' in st.session_state:
        del st.session_state['vs']

# Streamlit app
if __name__ == "__main__":
    st.title("Retrieval augmented generation")
    st.subheader("Plug in your data source")
    
    # Sidebar for selecting data source
    Data_Source = ["File", "Web Data"]
    option = st.sidebar.selectbox("Data Source", Data_Source, index=0)
    
    # File upload section
    if option == "File":
        uploaded_file = st.file_uploader("Upload a file:", type=["txt", "docx", "pdf"])
        if uploaded_file:
            add_data = st.button('Add Data', on_click=clear_vector_store)
            if add_data:
                with st.spinner('Reading and processing file ...'):
                    bytes_data = uploaded_file.read()
                    file_name = os.path.join('./', uploaded_file.name)
                    with open(file_name, 'wb') as f:
                        f.write(bytes_data)
                    #delete_pinecone_index(index_name='project')
                    data = load_documents(file_name)
                    chunked_data = chunk_data(data, chunk_size=1000)
                    vector_store = create_embeddings_vectorstore(chunked_data)
                    st.session_state.vs = vector_store
                    st.success('File is uploaded and can now be queried.')

    # Web data input section
    elif option == "Web Data":
        address = st.text_input('Please input the web address:')
        add_data = st.button('Scrape Data',on_click=clear_vector_store)
        if address:
            if add_data:
                with st.spinner('Reading and processing web address ...'):
                    delete_pinecone_index(index_name='project')
                    data = load_external(address)
                    chunked_data = chunk_data(data, chunk_size=1000)
                    vector_store = create_embeddings_vectorstore(chunked_data)
                    st.session_state.vs = vector_store
                    st.success('Text from the site has been scraped and can now be queried.')

    st.divider()

    # Text input for user's question
    question = st.text_input('Ask a question about the content of your file:')
    if question:
        if 'vs' in st.session_state:
            vector_store = st.session_state.vs
            answer = questions_answer(question, vector_store)
            st.text_area('Context Answer:', value=answer)
