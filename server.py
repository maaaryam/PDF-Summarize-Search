# grpc_server.py
import grpc
from concurrent import futures
import chatbot_pb2
import chatbot_pb2_grpc
from LLMmodel import load_pdf_from_path 
from langchain_community.document_loaders import PyPDFLoader
import os

class ChatbotService(chatbot_pb2_grpc.ChatbotServiceServicer):
 
    def SendPDFPaths(self, request, context):
        reply_status=[]
        # Here you could process the received PDF paths
        for pdf_path in request.pdf_paths:
            # Verify that the file exists
            if os.path.isfile(pdf_path):
                path = os.path.join(os.path.dirname(pdf_path), pdf_path)
                print(f"Received PDF path: {path}")
                loader = PyPDFLoader(os.path.join(os.path.dirname(pdf_path), pdf_path))

                try:
                        document = loader.load()
                        print(document)
                        reply_status.append('Success')
                except Exception as e:
                        reply_status.append(f'Failed to load: {e}')
            else:
                reply_status.append('File not found')
                print(f'File not found: {pdf_path}')

        # For simplicity, return a confirmation message
        return chatbot_pb2.PDFPathsResponse(reply="Completed", status=reply_status)

    def SendMessage(self, request, context):
        # Here you'll call your actual chatbot model to get the response
        # For simplicity, we're just echoing back the received message
        response_text = f"You said: {request.message}"
        print(response_text)
        return chatbot_pb2.ChatResponse(reply=response_text)
    


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    chatbot_pb2_grpc.add_ChatbotServiceServicer_to_server(ChatbotService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print('gRPC server running on port 50051...')
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
