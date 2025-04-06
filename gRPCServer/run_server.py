# gRPCServer/run_server.py

import grpc
from concurrent import futures
import recommendation_pb2_grpc
from recommendation_service import RecommenderServicer
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    # 注册我们的 servicer
    recommendation_pb2_grpc.add_RecommenderServicer_to_server(
        RecommenderServicer(),
        server
    )
    # 监听端口
    server.add_insecure_port('[::]:50051')
    server.start()
    print("gRPC server listening at port 50051")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
