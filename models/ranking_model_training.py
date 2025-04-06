# recommendation_system/ranking_model_training.py

import os
import xgboost as xgb
import numpy as np
from typing import List, Dict, Tuple


class RankingModelTrainer:
    """
    离线训练 Learning-to-Rank 模型(示例: XGBoost Ranker).
    其中 'label' 表示用户对该题目的“相关度”或“点击完成情况”:
      - label=1 表示正反馈 (点击、完成)
      - label=0 表示无反馈
    若数据中无 label 字段, 需要以别的字段(如 rating>2)推断.
    """

    def __init__(self):
        # 超参数可根据需要调整
        self.params = {
            "objective": "rank:pairwise",  # 排序目标
            "eval_metric": "ndcg",
            "eta": 0.1,
            "max_depth": 6,
            "seed": 42
        }
        self.num_boost_round = 50
        self.model = None

    def build_dataset(self, interactions: List[Dict], user_profile, question_db) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
        """
        构建 (X, y, group) 用于 XGBoost Ranker.
        interactions: 每条记录至少包含 {user_id, question_id, ...}
                      并根据 label或rating推断 label(是否点击/完成).

        user_profile: 访问 user_profile[user_id], 可能含 user_embedding, tag_weights, ...
        question_db:  访问 question_db[qid], 可能含 question["vector"], question["hot_score"], CFscore 等

        group: rank的分组. 这里可按 user_id 分组 => 对每个用户的一批 (question) 做排序
        """
        from collections import defaultdict
        user2records = defaultdict(list)
        for row in interactions:
            uid = row["user_id"]
            user2records[uid].append(row)

        X_list = []
        y_list = []
        group_list = []

        # 每个user是一组, group_size = len(该user的recs)
        for uid, recs in user2records.items():
            group_size = len(recs)
            group_list.append(group_size)

            # 取 user embedding, tag_weights
            profile = user_profile.get_user_profile(uid)
            user_emb = None
            tag_weights = {}
            if profile:
                user_emb = profile.get("embedding", None)
                tag_weights = profile.get("tag_weights", {})

            for row in recs:
                qid = row["question_id"]
                # 如果有 label 列, 则使用; 若无 => 尝试从 rating>2 推断
                label = row.get("label", None)
                if label is None:
                    # fallback: interpret label from rating
                    # rating>2 => label=1, else=0
                    rating_val = row.get("rating", 0)
                    if isinstance(rating_val, str):
                        rating_val = float(rating_val)
                    label = 1 if rating_val > 2 else 0

                qinfo = question_db.get(qid, {})
                # 构造特征
                feat = self.extract_features(uid, qid, user_emb, tag_weights, qinfo)
                X_list.append(feat)
                y_list.append(label)

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)
        group = np.array(group_list, dtype=np.int32)

        return X, y, group

    def extract_features(self, user_id, qid, user_emb, tag_weights, qinfo):
        """
        返回一个 特征向量 list/np.array
        可包含:
          - CF分数
          - 标签匹配度
          - user_emb与qinfo["vector"]余弦相似
          - qinfo["hot_score"]
          - qinfo["views"], qinfo["rating"]
          ...
        """
        feats = []

        # 1) CF分数(如 question_db 中存了 "cf_score" 或可动态计算)
        cf_score = qinfo.get("cf_score", 0.0)
        feats.append(cf_score)

        # 2) 标签匹配度( sum of user tag_weights on qinfo["tags"] )
        tag_sum = 0.0
        q_tags = qinfo.get("tags", [])
        for t in q_tags:
            tag_sum += tag_weights.get(t, 0.0)
        feats.append(tag_sum)

        # 3) user_emb ~ item_emb 余弦相似
        item_emb = qinfo.get("vector", None)
        sim = 0.0
        if user_emb is not None and item_emb is not None:
            if len(user_emb) > 0 and len(item_emb) > 0:
                dot = np.dot(user_emb, item_emb)
                norm_u = np.linalg.norm(user_emb)
                norm_i = np.linalg.norm(item_emb)
                if norm_u > 1e-9 and norm_i > 1e-9:
                    sim = dot / (norm_u * norm_i)
        feats.append(sim)

        # 4) hot_score, rating, views
        hot_score = qinfo.get("hot_score", 0.0)
        rating = qinfo.get("rating", 0.0)
        views = qinfo.get("views", 0.0)
        feats.append(hot_score)
        feats.append(rating)
        feats.append(views)

        return feats

    def train(self, X, y, group):
        """
        用 XGBoost Ranker 训练
        """
        dtrain = xgb.DMatrix(X, label=y)
        dtrain.set_group(group.tolist())

        self.model = xgb.train(self.params, dtrain, self.num_boost_round)
        print("[RankingModelTrainer] Training done with XGBoost ranker.")

    def save_model(self, model_path="ranker_xgb.model"):
        if self.model:
            self.model.save_model(model_path)
            print(f"[RankingModelTrainer] Saved model to {model_path}")


def demo_ranking_training():
    """
    演示:
      1) 构建 interactions => (user_id, question_id, rating, [label?])
      2) load user_profile, question_db
      3) build dataset => train => save
    """
    import csv

    # 假设 "data.csv" 里有 user_id, question_id, rating, (可选label)
    # 如果没有 label 字段, 就自动根据 rating>2 => label=1
    interactions = []
    with open("data.csv", 'r', encoding='utf-8') as f:
        rd = csv.DictReader(f)
        for row in rd:
            uid = int(row["user_id"])
            qid = str(row["question_id"])
            r = row.get("rating", "0")
            try:
                rating_val = float(r)
            except:
                rating_val = 0.0
            # 读 label
            if "label" in row:
                label_val = int(row["label"])
            else:
                label_val = None  # 待后面自动判断

            interactions.append({
                "user_id": uid,
                "question_id": qid,
                "rating": rating_val,
                "label": label_val
            })

    # 2) load user_profile => must have user_emb, tag_weights
    #   你需要先运行  some CF or MF to store user_emb, or ...
    from user_profile import UserProfileManager
    upm = UserProfileManager()
    # upm.load_xxx()  # 省略, 这里假设 user_profile 已经有 embedding, tag_weights

    # 3) question_db => {qid: {...}}, 里面最好包含 item_emb, hot_score, rating, views, ...
    question_db = {}
    # TODO: load or pass real data

    # 4) build
    trainer = RankingModelTrainer()
    X, y, group = trainer.build_dataset(interactions, upm, question_db)

    # 5) train
    trainer.train(X, y, group)

    # 6) save
    trainer.save_model("ranker_xgb.model")
    print("[demo_ranking_training] => done.")


if __name__ == "__main__":
    demo_ranking_training()
