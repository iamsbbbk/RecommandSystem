# test_client.py
import grpc
import recommendation_pb2
import recommendation_pb2_grpc

def test_recommend():
    channel = grpc.insecure_channel('localhost:50051')
    stub = recommendation_pb2_grpc.RecommenderStub(channel)

    # 1) SendInteraction
    request_int = recommendation_pb2.InteractionRequest()
    inter = request_int.interactions.add()
    inter.user_base.user_id = 123
    inter.user_base.interest_tags.extend(["dp","graph"])
    inter.question_id = "496"
    inter.timestamp = "2023-09-25T10:20:00"
    inter.rating = 4.0
    inter.views = 1

    resp_int = stub.SendInteraction(request_int)
    print("SendInteraction resp:", resp_int.success, resp_int.message)

    # 2) Recommend
    req_rec = recommendation_pb2.RecommendRequest()
    req_rec.user_base.user_id = 123
    req_rec.user_base.interest_tags.extend(["dp","graph"])
    req_rec.question_id = "496"
    req_rec.rating = 4.0
    req_rec.views = 1
    req_rec.top_n = 2
    resp_rec = stub.Recommend(req_rec)

    print("Recommend results:")
    for item in resp_rec.results:
        print(f"QID={item.question_id}, tags={list(item.tags)}, rating={item.rating}, hot={item.hot_score}")

if __name__=="__main__":
    test_recommend()
