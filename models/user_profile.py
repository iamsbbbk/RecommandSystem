# models/user_profile.py

import os
import csv
import statistics
import numpy as np
from typing import Dict, List, Set

class UserProfileManager:

    def __init__(self):
        # user_profiles 数据结构示例：
        # {
        #   user_id1: {
        #       "items": [...],
        #       "embedding": np.array([...]),
        #       "tag_weights": {"贪心算法": 2.0, "排序算法": 1.0, ...},
        #       ...
        #   },
        #   user_id2: {...}
        # }
        self.user_profiles: Dict[int, Dict] = {}

        # 这三个权重用来计算 raw_score，如果 CSV 中没有 is_fav / correct_rate，可以忽略
        self.weight_rating = 0.5
        self.weight_fav = 0.3
        self.weight_correct = 0.2

    def load_user_interactions(self, csv_path: str):
        """
        读取 CSV 并初始化 user_profiles。包括：
          1) 计算 raw_score（基于 rating、fav、correct_rate）
          2) 从 user_interest 字段解析用户的兴趣标签，并更新到 tag_weights
        """
        if not os.path.exists(csv_path):
            print(f"[UserProfileManager] CSV {csv_path} not found.")
            return

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    uid = int(row.get("user_id", "0"))
                    qid = row.get("question_id", "").strip()
                except:
                    continue

                # 跳过无效的 user_id 或 question_id
                if not qid or uid == 0:
                    continue

                # 读取并计算交互分数
                r = float(row.get("rating", "0"))
                fav = float(row.get("is_fav", "0"))
                corr = float(row.get("correct_rate", "0"))
                raw_score = (self.weight_rating * r
                             + self.weight_fav * fav
                             + self.weight_correct * corr)

                # 初始化用户画像
                if uid not in self.user_profiles:
                    self.user_profiles[uid] = {
                        "items": [],
                        "embedding": np.array([], dtype=np.float32),
                        "tag_weights": {},
                        "done_questions": set(),       # 若需要记录做题信息
                        "recently_recommended": set()  # 若需要记录最近推荐信息
                    }

                # 记录当前交互信息到 items
                self.user_profiles[uid]["items"].append({
                    "question_id": qid,
                    "raw_score": raw_score
                })

                # 读取 user_interest 并更新 tag_weights
                user_interest_str = row.get("user_interest", "").strip()
                if user_interest_str:
                    interest_tags = user_interest_str.split()
                    for tag in interest_tags:
                        # 这里简单地每遇到一次就 +1，也可以根据 rating 等因素加权
                        self.user_profiles[uid]["tag_weights"][tag] = \
                            self.user_profiles[uid]["tag_weights"].get(tag, 0.0) + 1.0

        # （可选）对所有用户的 tag_weights 做归一化处理
        self._normalize_tag_weights()

    def _normalize_tag_weights(self):
        """
        可选：对 tag_weights 进行归一化（如总和归一化），
        防止有些用户出现非常大的标签累加值。
        """
        for uid, up in self.user_profiles.items():
            tw = up["tag_weights"]
            total = sum(tw.values())
            if total > 1e-9:
                for t in tw:
                    tw[t] /= total

    def normalize_scores_zscore(self):
        """
        对 items 中的 raw_score 做 Z-score 归一化。
        """
        for uid, up in self.user_profiles.items():
            items = up["items"]
            if not items:
                continue
            scores = [x["raw_score"] for x in items]
            meanv = statistics.mean(scores)
            stdev = statistics.pstdev(scores)
            if stdev < 1e-9:
                for x in items:
                    x["normalized_score"] = 0.0
            else:
                for x in items:
                    x["normalized_score"] = (x["raw_score"] - meanv) / stdev

    def get_user_item_scores(self, user_id) -> List[tuple]:
        """
        返回 (question_id, normalized_score) 列表
        """
        if user_id not in self.user_profiles:
            return []
        items = self.user_profiles[user_id]["items"]
        return [(x["question_id"], x.get("normalized_score", 0.0))
                for x in items]

    def list_all_users(self):
        return list(self.user_profiles.keys())

    def get_user_profile(self, user_id: int) -> Dict:
        """
        返回某个用户画像的完整字典，包括 items、embedding、tag_weights 等
        """
        if user_id not in self.user_profiles:
            return None
        return self.user_profiles[user_id]


def demo_profile_enhancement():
    """
    简单演示：读取 CSV，构建用户画像，检查结果
    """
    upm = UserProfileManager()

    upm.load_user_interactions("data.csv")
    upm.normalize_scores_zscore()

    all_users = upm.list_all_users()
    if not all_users:
        return

    sample_uid = all_users[1]
    items = upm.get_user_item_scores(sample_uid)
    print(f"[demo] user_id={sample_uid}, item_scores => {items[:10]}")

    prof = upm.get_user_profile(sample_uid)
    print(f"[demo] user_id={sample_uid}, embedding =>", prof["embedding"])
    print(f"[demo] user_id={sample_uid}, tag_weights =>", prof["tag_weights"])


if __name__ == "__main__":
    demo_profile_enhancement()
