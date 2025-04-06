# gRPCServer/recommendation_service.py

import os
import json
import recommendation_pb2
import recommendation_pb2_grpc

from models.recommender import RecommenderService
from models.user_profile import UserProfileManager
from models.collaborative_filter import CollaborativeFilterModel

import numpy as np


class RecommenderServicer(recommendation_pb2_grpc.RecommenderServicer):
    """
    实现 proto 中定义的 Recommender 服务 (SendInteraction + Recommend)
    """

    def __init__(self):
        # 1) 初始化 UserProfileManager
        self.profile_mgr = UserProfileManager()

        # 2) 初始化 RecommenderService，并传入用户画像管理
        self.recommender_service = RecommenderService(self.profile_mgr)

        # 3) 初始化 CF 模型 (协同过滤)
        self.cf_model = CollaborativeFilterModel(sim_threshold=0.0)
        self.recommender_service.load_cf_model(self.cf_model)

        # 4) 从 JSON 文件加载题库数据（返回列表）
        json_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "models",
                                 "question_bank.json")
        question_list = self.load_question_bank_from_json(json_path)
        self.recommender_service.load_question_db(question_list)

        print("[RecommenderServicer] Initialized all components")

    def load_question_bank_from_json(self, json_file):
        if not os.path.exists(json_file):
            print(f"[RecommenderServicer] JSON file {json_file} not found!")
            return []
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                # 直接返回题库列表
                question_list = json.load(f)
            print(f"[RecommenderServicer] Loaded question bank from JSON with {len(question_list)} questions.")
            return question_list
        except Exception as e:
            print(f"[RecommenderServicer] Error loading JSON: {e}")
            return []

    def SendInteraction(self, request, context):
        """
        gRPC 方法: SendInteraction(InteractionRequest) -> InteractionResponse
        处理用户上传的一批交互数据
        """
        total = 0
        new_interactions = []
        for interaction in request.interactions:
            user_id = interaction.user_base.user_id
            if user_id not in self.profile_mgr.user_profiles:
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

            # 更新用户画像中的已做题目
            self.profile_mgr.user_profiles[user_id]["done_questions"].add(qid)

            # 收集 (user_id, question_id, rating) 作为协同过滤交互数据
            new_interactions.append((user_id, qid, rating))
            total += 1

        # 合并旧的交互数据，并 refit CF 模型（示例逻辑）
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
        gRPC 方法: Recommend(RecommendRequest) -> RecommendResponse
        返回推荐结果
        """
        user_id = request.user_base.user_id
        if user_id not in self.profile_mgr.user_profiles:
            self.profile_mgr.user_profiles[user_id] = {
                "items": [],
                "embedding": np.array([], dtype=np.float32),
                "tag_weights": {},
                "done_questions": set(),
                "recently_recommended": set()
            }
        # 获取请求参数
        qid = request.question_id
        rating_val = request.rating
        views_val = request.views
        top_n = request.top_n if request.top_n > 0 else 5

        rec_list = self.recommender_service.recommend(user_id, top_n=top_n)

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
