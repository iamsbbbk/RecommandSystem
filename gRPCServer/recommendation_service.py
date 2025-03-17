# recommendation_service.py

import recommendation_pb2
import recommendation_pb2_grpc

from models.recommender import RecommenderService
from models.user_profile import UserProfileManager
from models.collaborative_filter import CollaborativeFilterModel

import numpy as np


class RecommenderServicer(recommendation_pb2_grpc.RecommenderServicer):
    """
    继承并实现 proto 中的 Recommender 服务
    (SendInteraction + Recommend)
    """

    def __init__(self):
        # 1) 初始化UserProfileManager
        self.profile_mgr = UserProfileManager()
        # 2) 初始化 RecommenderService
        self.recommender_service = RecommenderService(self.profile_mgr)
        # 3) 初始化 CF model
        self.cf_model = CollaborativeFilterModel(sim_threshold=0.0)
        #   这里先不fit任何数据, 等有数据或需要时再 fit
        self.recommender_service.load_cf_model(self.cf_model)

        # 你可能还需要 question_db, hot_list 等
        self.question_db = {}
        # 也可以由外部传入, 或者此处直接设置
        # self.recommender_service.load_question_db(...)

        print("[RecommenderServicer] Initialized all components")

    def SendInteraction(self, request, context):
        """
        gRPC方法: SendInteraction(InteractionRequest) -> InteractionResponse
        负责处理用户上传的一批交互(UserInteraction)
        """
        total = 0
        new_interactions = []
        for interaction in request.interactions:
            user_id = interaction.user_base.user_id
            if user_id not in self.profile_mgr.user_profiles:
                # 若没有该用户, 初始化
                self.profile_mgr.user_profiles[user_id] = {
                    "items": [],
                    "embedding": np.array([], dtype=np.float32),
                    "tag_weights": {},
                    "done_questions": set(),
                    "recently_recommended": set()
                }

            qid = interaction.question_id
            rating = interaction.rating
            views = interaction.views
            timestamp = interaction.timestamp
            # interest_tags = list(interaction.user_base.interest_tags)
            # context_str = interaction.user_base.context

            # 1) update user_profile "done_questions"
            self.profile_mgr.user_profiles[user_id]["done_questions"].add(qid)

            # 2) 也许你要写tag_weights 或 embedding, 这里不多做, 取决于业务

            # 3) 收集( user_id, qid, rating ) 用于协同过滤
            #    你可以存储(views, timestamp) 如果有需要
            new_interactions.append((user_id, qid, rating))
            total += 1

        # 将 old_data + new_interactions => refit cf model (示例)
        old_data = []
        for uid, items_map in self.cf_model.user_items.items():
            for item_id, r in items_map.items():
                old_data.append((uid, item_id, r))
        merged = old_data + new_interactions

        self.cf_model.fit(merged)
        self.recommender_service.load_cf_model(self.cf_model)

        resp = recommendation_pb2.InteractionResponse(
            success=True,
            message=f"Processed {total} interactions."
        )
        return resp

    def Recommend(self, request, context):
        """
        gRPC方法: Recommend(RecommendRequest) -> RecommendResponse
        """
        user_id = request.user_base.user_id
        if user_id not in self.profile_mgr.user_profiles:
            # 如果用户画像不存在, 创建一个简单的
            self.profile_mgr.user_profiles[user_id] = {
                "items": [],
                "embedding": np.array([], dtype=np.float32),
                "tag_weights": {},
                "done_questions": set(),
                "recently_recommended": set()
            }

        # question_id, rating, views, top_n
        qid = request.question_id
        rating_val = request.rating
        views_val = request.views
        top_n = request.top_n if request.top_n > 0 else 5

        # 这里可将 (user_id, qid, rating_val) 视为新的行为(若需要)
        # 也可不立即更新 CF, 以免太频繁
        # self.cf_model.fit(...)

        # 调用 recommend
        rec_list = self.recommender_service.recommend(user_id, top_n=top_n)

        # 组装 RecommendResponse
        resp = recommendation_pb2.RecommendResponse()
        for rec_qid in rec_list:
            qinfo = self.recommender_service.question_db.get(rec_qid, {})
            item = resp.results.add()
            item.question_id = rec_qid
            if "tags" in qinfo:
                item.tags.extend(qinfo["tags"])
            item.rating = qinfo.get("rating", 0.0)
            item.views = qinfo.get("views", 0)
            item.hot_score = qinfo.get("hot_score", 0.0)
        return resp
