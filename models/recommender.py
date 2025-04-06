# models/recommender.py

import random
import math
from typing import List, Dict

from models.collaborative_filter import CollaborativeFilterModel
from models.diversity import diversity_rerank

class RecommenderService:
    """
    在线推荐核心逻辑:
    - 多通道召回(标签 / 用户兴趣 / CF / 热门 + 随机探索)
    - 过滤: 去除已做/已推荐
    - 排序融合(可加随机加权)
    - 多样性重排
    """

    def __init__(self, user_profile_manager):
        self.user_profile_manager = user_profile_manager
        self.question_db: Dict[str, dict] = {}
        self.hot_questions: List[dict] = []
        self.question2vec: Dict[str, list] = {}
        self.cf_model = None

    def load_question_db(self, question_list: List[dict]):
        self.question_db.clear()
        self.question2vec.clear()
        for q in question_list:
            qid = str(q["id"])
            self.question_db[qid] = q
            if "vector" in q:
                self.question2vec[qid] = q["vector"]
            else:
                self.question2vec[qid] = []
        print(f"[RecommenderService] Loaded {len(question_list)} questions into question_db.")

    def update_hot_list(self, hot_list: List[dict]):
        self.hot_questions = hot_list
        for item in hot_list:
            qid = str(item.get("id",""))
            sc = item.get("hot_score",0.0)*0.1
            if qid in self.question_db:
                self.question_db[qid]["hot_score"] = sc
        print(f"[RecommenderService] Updated hot list, size={len(hot_list)}")

    def load_cf_model(self, cf_model: CollaborativeFilterModel):
        self.cf_model = cf_model
        print("[RecommenderService] CF model loaded for user-based or item-based CF recall.")

    # =========== 多通道召回 ===========

    def recall_by_tags(self, user_id, top_tag_count=3):
        profile = self.user_profile_manager.get_user_profile(user_id)
        if not profile:
            return []
        tag_weights = profile.get("tag_weights", {})
        # 取前 n 个标签
        top_tags = sorted(tag_weights.items(), key=lambda x: x[1], reverse=True)[:top_tag_count]

        candidate_qids = set()
        for (tag, weight) in top_tags:
            for q_id, q_info in self.question_db.items():
                if tag in q_info.get("tags", []):
                    candidate_qids.add(q_id)
        return list(candidate_qids)

    def recall_by_user_interest(self, user_id):
        profile = self.user_profile_manager.get_user_profile(user_id)
        if not profile:
            return []
        interest_str = profile.get("user_interest","").strip()
        if not interest_str:
            return []
        interests = interest_str.split()
        candidate_qids = set()
        for q_id, qinfo in self.question_db.items():
            q_tags = qinfo.get("tags", [])
            if any(i in q_tags for i in interests):
                candidate_qids.add(q_id)
        return list(candidate_qids)

    def recall_by_cf(self, user_id, top_k=10):
        if not self.cf_model:
            return []
        return self.cf_model.predict_k_items_for_user(user_id, top_k=top_k)

    def recall_hot_questions(self, limit=10):
        return [ str(q["id"]) for q in self.hot_questions[:limit] ]

    def recall_random_explore(self, limit=10):
        # 随机挑选 question_db 的若干项
        all_qids = list(self.question_db.keys())
        if len(all_qids)<=limit:
            return all_qids
        return random.sample(all_qids, limit)

    def recall_candidates(self, user_id):
        """
        多通道: tags + interest + cf + hot + random => union
        """
        tag_set = set(self.recall_by_tags(user_id, top_tag_count=3))
        interest_set = set(self.recall_by_user_interest(user_id))
        cf_set = set(self.recall_by_cf(user_id, top_k=10))
        hot_set = set(self.recall_hot_questions(limit=5))
        random_set = set(self.recall_random_explore(limit=5))

        merged = tag_set.union(interest_set).union(cf_set).union(hot_set).union(random_set)
        return list(merged)

    # =========== 过滤 & 排序 & 多样性 ===========

    def filter_candidates(self, user_id, candidate_qids):
        profile = self.user_profile_manager.get_user_profile(user_id)
        if not profile:
            return []
        done_set = profile.get("done_questions", set())
        # recently_recommended: 在profile中记录, 用来避免短期内重复
        rec_set = profile.get("recently_recommended", set())
        filtered = [q for q in candidate_qids if (q not in done_set) and (q not in rec_set)]
        return filtered

    def rank_and_sort(self, user_id, candidate_qids, top_n=10,
                      scheme="A", use_diversity=True, alpha=0.5):
        # alpha调大 => 更强多样性
        profile = self.user_profile_manager.get_user_profile(user_id)
        if not profile:
            return []

        # 1) 不同融合方案
        schemes = {
            "A": {"tag": 0.4, "cf": 0.3, "hot": 0.15, "rating": 0.1, "views": 0.05},
            "B": {"tag": 0.3, "cf": 0.4, "hot": 0.2,  "rating": 0.05,"views":0.05},
        }
        if scheme not in schemes:
            scheme = "B"
        w = schemes[scheme]

        tag_weights = profile.get("tag_weights", {})
        results = []
        for qid in candidate_qids:
            qinfo = self.question_db.get(qid,{})
            # tag
            sum_tag = 0.0
            for t in qinfo.get("tags",[]):
                sum_tag += tag_weights.get(t,0.0)
            # cf
            cf_score = 0.0
            if self.cf_model:
                cf_score = self.cf_model.predict_score(user_id, qid)
            # hot
            hot_val = qinfo.get("hot_score",0.0)*0.1
            # rating & views
            rat = qinfo.get("rating",0)
            vws = qinfo.get("views",0)

            base_score = (w["tag"]*sum_tag + w["cf"]*cf_score
                          + w["hot"]*hot_val + w["rating"]*rat + w["views"]*vws)
            # 随机微扰, 让结果更有新鲜感
            base_score += random.random()*0.01

            # debug:
            # print(f"[DebugRank] QID={qid}, sum_tag={sum_tag}, cf={cf_score}, hot={hot_val}, rating={rat}, views={vws}, => {base_score:.2f}")

            results.append((qid, base_score))

        # 排序
        results.sort(key=lambda x:x[1], reverse=True)

        # 先取更大 topN', 然后多样性
        big_topN = min(len(results), top_n*3)  # 取3倍, 留更多给多样性 re-rank
        top_candidates = [x[0] for x in results[:big_topN]]

        if use_diversity:
            final_list = diversity_rerank(top_candidates, self.question_db, top_n=top_n, alpha=alpha)
        else:
            final_list = top_candidates[:top_n]
        return final_list

    def recommend(self, user_id, top_n=5):
        cands = self.recall_candidates(user_id)
        cands = self.filter_candidates(user_id, cands)
        final_list = self.rank_and_sort(user_id, cands, top_n=top_n, scheme="A", use_diversity=True, alpha=0.5)

        # 把 final_list 加入 user_profile["recently_recommended"] 以免短期内重复
        profile = self.user_profile_manager.get_user_profile(user_id)
        if profile is not None:
            if "recently_recommended" not in profile:
                profile["recently_recommended"] = set()
            for qid in final_list:
                profile["recently_recommended"].add(qid)
            # 可再限制 recently_recommended size
            if len(profile["recently_recommended"])>50:
                # 随机移除一些
                profile["recently_recommended"] = set(list(profile["recently_recommended"])[-50:])

        return final_list
