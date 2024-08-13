from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.chains.summarize import load_summarize_chain
# from langchain.llms import HuggingFacePipeline
from transformers import pipeline
import os
import pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
# from langchain.llms import HuggingFaceHub
from langchain_experimental.text_splitter import SemanticChunker
from dotenv import load_dotenv
from langchain_community.llms import HuggingFacePipeline
from langchain_community.llms import HuggingFaceHub

 

load_dotenv()
os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.getenv('HUGGINGFACEHUB_API_TOKEN')
os.environ['PINECONE_API_KEY'] = os.getenv('api_key')



##LOAD File
def load_pdf_from_path(pdf_path):
    """
    Load a PDF document using PyPDFLoader from a given file path.
    """
    loader = PyPDFDirectoryLoader(pdf_path)
    document = loader.load()
    return document


llm_embedding = HuggingFaceEmbeddings(model_name ="sentence-transformers/all-MiniLM-L6-v2")
##SPLIT DOCUMENTS
def split_document_into_chunks(document, chunk_size=500, chunk_overlap=20):
    text_splitter = SemanticChunker(llm_embedding)
    chunks = text_splitter.split_documents(document)
    return chunks


##SUMMARIZATION
def summarize_chunks(documents):

    summarization_pipeline = pipeline(
    task="summarization",
    model="facebook/bart-large-cnn",
    tokenizer="facebook/bart-large-cnn",
    max_length=100,
    min_length=10,)

    llm_summarize = HuggingFacePipeline(pipeline=summarization_pipeline)

    for doc in documents:
        chunks = split_document_into_chunks([doc])
        summarize_chain = load_summarize_chain(llm_summarize, chain_type="map_reduce")
        summary = summarize_chain.run(chunks)
        print(f"Source Document:  {doc.metadata['source']}")
        print("Summary:")
        print(summary)
        print("\n" + "="*80 + "\n")

    return 



##STORE IN VECTORDB
llm_embedding = HuggingFaceEmbeddings(model_name ="sentence-transformers/all-MiniLM-L6-v2")


#Initialize Pinecone
pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
index_name="projectinternship"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws',region='us-west-2')
    )


#STORE DOCUMENTS IN VECTORDB
def store_documents_in_vector_db(documents, index_name="projectinternship"):

    vectorstore = PineconeVectorStore.from_documents(
        split_document_into_chunks(documents),
        llm_embedding,
        index_name=index_name
    )
    return vectorstore


#SEARCH IN PDFS
def setup_retrieval_chain(vectorstore):
    llm_generation = HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        huggingfacehub_api_token=os.getenv('HUGGINGFACEHUB_API_TOKEN')
    )
    
    retriever = vectorstore.as_retriever()
    prompt_search_query = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up to get information relevant to the conversation")
    ])
    
    retriever_chain = create_history_aware_retriever(llm_generation, retriever, prompt_search_query)
    prompt_get_answer = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\\n\\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    
    document_chain = create_stuff_documents_chain(llm_generation, prompt_get_answer)
    retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)
    
    return retrieval_chain



pdf_path = pdf_path = "C:/Users/asus/Downloads/llm_project/LLM/q10chuwvg6.pdf"
documents = load_pdf_from_path(pdf_path)
summary = summarize_chunks(documents)
vectorstore = store_documents_in_vector_db(documents)
retrieval_chain = setup_retrieval_chain(vectorstore)


