import streamlit as st
import grpc
import chatbot_pb2
import chatbot_pb2_grpc

# Initialize gRPC channel and stub
channel = grpc.insecure_channel('localhost:50051')
stub = chatbot_pb2_grpc.ChatbotServiceStub(channel)

def send_pdf_paths(pdf_paths):
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = chatbot_pb2_grpc.ChatbotServiceStub(channel)
        response = stub.SendPDFPaths(chatbot_pb2.PDFPathsRequest(pdf_paths=pdf_paths))
    return response.reply , response.status


if 'chat_enabled' not in st.session_state:
    st.session_state['chat_enabled'] = False
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

st.title('Chatbot Interface')

pdf_files = st.file_uploader("Choose PDF files", accept_multiple_files=True ,type=['pdf'])

if st.button('Send PDFs') and pdf_files:
    pdf_paths = [pdf.name for pdf in pdf_files]
    reply_message,reply_status = send_pdf_paths(pdf_paths)
    # reply_message = response_message.reply  # This holds the 'Completed' message or failure message
    # reply_status = response_message.status  # This holds the status for each individual file
    if reply_message == "Completed":
        st.session_state.chat_enabled = True
        st.success("PDFs sent successfully. You can now chat with the server.")
        for status in reply_status:
            # Use st.write, st.info, st.success, st.warning, etc., to display status messages.
            st.text(status)
    else:
        st.session_state.chat_enabled = False
        st.error("Failed to send PDFs.")

def send_message():
    if st.session_state.user_input:
        user_message = st.session_state.user_input
        response = stub.SendMessage(chatbot_pb2.ChatRequest(message=user_message))
        st.session_state.chat_history.append(("You", user_message))
        st.session_state.chat_history.append(("Chatbot", response.reply))
        st.session_state.user_input = ""  # Reset the message text box after sending the message

# The chat input box
if st.session_state.chat_enabled:
    user_input = st.text_input("Enter your message:", key="user_input", on_change=send_message)

    # The send button
    st.button("Send", on_click=send_message)

# Display chat history
st.write("Chat History:")
for who, line in st.session_state.chat_history:
    st.text(f"{who}: {line}")
