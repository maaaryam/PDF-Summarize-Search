# Python virtual environment
```
python -m venv chatbot-env
chatbot-env\Scripts\activate
```
Use cmd if not working in vscode terminal


# PDF Search System with LLM

This project demonstrates a system that leverages language models for summarizing and building a document-based question-answering system using LangChain, Pinecone, and Huggingface models.It uses sentence transformers for embeddings, vector databases for storing document representations, and large language models (LLMs) for generating natural language queries based on chat history.

## Features

- Load and process PDF documents from file paths.
- Efficiently handle long documents by splitting them into semantic chunks.
- Leverage a summarization Huggingface model to create concise summaries of document contents.
- Use Pinecone, a vector database, for storing and searching within document embeddings.
- Implement RAG for intelligent, context-aware search based on user-provided chat history.

## Setup


### Install required packages

Before running the code, ensure you have the necessary dependencies installed and API keys set up.

```
pip install -q langchain pypdf -U langchain-community sentence-transformers langchain_pinecone pinecone-client langchain_experimental python-dotenv
```
### API Keys

This system requires API keys for Pinecone and Hugging Face. Save your API keys to .env file.

```
HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_token_here
api_key=your_pinecone_api_key_here
```

## Usage

Run the code to process a PDF file, summarize its contents, and set up the retrieval chain:

```
pdf_path = "your_pdf_file_path_here"
document = load_pdf_from_path(pdf_path)
for doc in document:
    summary = summarize_chunks([doc])
vectorstore = store_documents_in_vector_db(document)
retrieval_chain = setup_retrieval_chain(vectorstore)
```



# Chatbot with gRPC Integration
This project consists of a simple chatbot service implemented using gRPC for server-client communication. The chatbot allows users to upload PDF files and send them to the server.

## Features
### Uploading PDFs:

- Use the file uploader in the Streamlit interface to select one or more PDF files.
- Click the "Send PDFs" button to send the selected PDFs to the gRPC server.
- The server processes the PDFs, and upon successful upload, the chat feature is enabled.

### Chatting with the Server:

- After successfully uploading PDFs, you can start a conversation with the server.
- Enter your message in the chat input box and click "Send" to  receive a response from the server.
- The conversation history will be displayed below the chat input.

## Setup 

### Install required packages:
```
pip install grpcio grpcio-tools streamlit
```
probably VPN is needed

### Generate APIs proto type for chatbot
```
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. chatbot.proto
```

## Usage 

### Run web app client
Open a cmd in project folder
```
streamlit run web_client.py
```

### run server
Open another cmd in prject folder
```
python server.py
```

