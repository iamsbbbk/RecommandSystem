# test_client.py

import grpc
import recommendation_pb2
import recommendation_pb2_grpc
import time


def run():
    # 创建到服务器的连接（确保服务器已经在 localhost:50051 启动）
    channel = grpc.insecure_channel('localhost:50051')
    stub = recommendation_pb2_grpc.RecommenderStub(channel)

    # 构造用户基础信息
    user_base = recommendation_pb2.UserBase(
        user_id=10,
        interest_tags=["模拟算法", "贪心算法"],
        context="test_session_001"
    )

    # 1. 上报用户交互数据：假设用户浏览并给题目 "1001" 进行了交互
    interaction = recommendation_pb2.UserInteraction(
        user_base=user_base,
        question_id="1001",
        timestamp="2025-04-06T10:23:42.931726",
        rating=4.0,
        views=1
    )
    interaction_request = recommendation_pb2.InteractionRequest(
        interactions=[interaction]
    )

    send_response = stub.SendInteraction(interaction_request)
    print("SendInteraction response:", send_response.message)

    # 暂停一小会儿，模拟服务端处理交互数据后再请求推荐
    time.sleep(1)

    # 2. 请求推荐：传入用户信息及最近的交互数据（这里使用同一题目 "1001" 作为参考）
    recommend_request = recommendation_pb2.RecommendRequest(
        user_base=user_base,
        question_id="1001",  # 最近交互的题目 ID
        rating=4.0,
        views=1,
        timestamp="2025-04-06T10:23:42.931726",
        top_n=5
    )

    recommend_response = stub.Recommend(recommend_request)

    print("\nRecommend response:")
    if not recommend_response.results:
        print("No recommendations found!")
    else:
        for qi in recommend_response.results:
            print(f"Question ID: {qi.question_id}")
            print(f"  Tags: {', '.join(qi.tags)}")
            print(f"  Rating: {qi.rating}")
            print(f"  Views: {qi.views}")
            print(f"  Hot Score: {qi.hot_score}")
            print("-" * 40)


if __name__ == "__main__":
    run()
