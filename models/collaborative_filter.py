# recommendation_system/collaborative_filter.py

import math
from collections import defaultdict

class CollaborativeFilterModel:
    """
    示例: Item-based协同过滤
    - fit() 通过用户-题目-评分 计算 item相似度矩阵 self.item_sim_map
    - predict_k_items_for_user() 给定 user_id 召回最相似item
    - predict_score(user_id, item_id) 计算用户对 item_id 的预测评分(用于最终打分)
    """

    def __init__(self, sim_threshold=0.0):
        self.sim_threshold = sim_threshold
        self.item_sim_map = defaultdict(dict)  # {itemA: {itemB: simAB, ...}, ...}
        self.user_items = defaultdict(dict)    # {user: {item: rating}}
        self.item_users = defaultdict(dict)    # {item: {user: rating}}

    def fit(self, interaction_rows):
        """
        interaction_rows: List of (user_id, item_id, rating)
        构建 user_items, item_users, 并计算 item相似度
        """
        # 1) 读取数据
        for (uid, qid, r) in interaction_rows:
            self.user_items[uid][qid] = r
            self.item_users[qid][uid] = r

        # 2) 计算 item两两相似度(余弦)
        items = list(self.item_users.keys())
        n = len(items)
        print(f"[CFModel] Building item-based similarity for {n} items...")

        # 先计算 norm
        item_norm = {}
        for it in items:
            ssum=0.0
            for u,rat in self.item_users[it].items():
                ssum += rat**2
            item_norm[it] = math.sqrt(ssum)

        # 构建相似度
        for i in range(n):
            for j in range(i+1, n):
                A = items[i]
                B = items[j]
                # dot = sum( rA*rB ) over all user that rated both
                dot = 0.0
                for userA, rA in self.item_users[A].items():
                    if userA in self.item_users[B]:
                        rB = self.item_users[B][userA]
                        dot += (rA * rB)
                normAB = item_norm[A]*item_norm[B]
                if normAB>0:
                    sim = dot/normAB
                else:
                    sim=0.0
                if sim > self.sim_threshold:
                    self.item_sim_map[A][B] = sim
                    self.item_sim_map[B][A] = sim

        print("[CFModel] item-based CF sim construction done.")

    def predict_k_items_for_user(self, user_id, top_k=10):
        """
        根据用户已做题, 给出最相似的 topK 未做题
        分数score(item)= sum(sim(item,i)*rating(user,i))
        """
        if user_id not in self.user_items:
            return []
        done_items = self.user_items[user_id]  # user对它们有评分
        # 计算对所有 未做过 item 的预测
        score_map = defaultdict(float)
        for done_it, ratDone in done_items.items():
            if done_it not in self.item_sim_map:
                continue
            for other_it, simAB in self.item_sim_map[done_it].items():
                if other_it in done_items:
                    continue
                # 累加
                score_map[other_it]+= simAB*ratDone

        # 排序
        sorted_candidates = sorted(score_map.items(), key=lambda x:x[1], reverse=True)
        topCands = [it for it,sc in sorted_candidates[:top_k]]
        return topCands

    def predict_score(self, user_id, item_id):
        """
        返回用户user_id对 item_id 的预测分, 用于recommend排序
        分数 = sum( sim(item_id, done_it)* rating(user, done_it) ) / sum_of_sim
        """
        if user_id not in self.user_items:
            return 0.0
        done_items = self.user_items[user_id]  # {item: rating}
        if item_id not in self.item_sim_map:
            return 0.0

        numerator=0.0
        denominator=0.0
        for done_it, ratingDone in done_items.items():
            simVal = self.item_sim_map[item_id].get(done_it, 0.0)

            if simVal>0:
                numerator += (simVal*ratingDone)
                denominator+= simVal
        if denominator>0:
            return numerator/denominator
        else:
            return 0.0