# recommendation_system/main.py

import os
import sys

from data_pipeline import DataPipeline
from offline_analysis import EnhancedContentAnalysis, EnhancedHotTopicAnalysis
from user_profile import UserProfileManager
from recommender import RecommenderService
from collaborative_filter import CollaborativeFilterModel
import numpy as np
def aggregate_users_and_questions(rows):
    """
    将 DataPipeline 的清洗后数据(rows) => users, questions 结构。
    users: [
      {
         "user_id": ...,
         "user_interest": "...",
         "questions": [ {"question_id":..., "rating":..., "views":...}, ... ]
      },
      ...
    ]
    questions: [
      {
        "id": str(question_id),
        "text": "题目描述",
        "tags": [...],
        "views": int(...),
        "rating": int(...),
        "timestamp": "...",
        ...
      },
      ...
    ]
    """
    from collections import defaultdict

    user_map = {}
    question_map = {}

    for row in rows:
        uid = row.get("user_id", 0)
        qid = row.get("question_id", 0)

        # 确保 uid 在 user_map 里
        if uid not in user_map:
            user_map[uid] = {
                "user_id": uid,
                "user_interest": row.get("user_interest","").strip(),
                "questions": []
            }
        # 如果 user_interest 为空且当前 row 有,则赋值
        if (not user_map[uid]["user_interest"]) and row.get("user_interest","").strip():
            user_map[uid]["user_interest"] = row["user_interest"].strip()

        # 累积 questions
        user_map[uid]["questions"].append({
            "question_id": qid,
            "rating": row.get("rating", 0),
            "views": row.get("views", 0)
        })

        # 构建 question_map
        if qid not in question_map:
            kw_str = row.get("question_keywords","").strip()
            kw_list = kw_str.split() if kw_str else []
            desc = row.get("question_description","").strip()
            question_map[qid] = {
                "id": str(qid),
                "text": desc,
                "tags": kw_list,
                "views": int(row.get("views",0)),
                "rating": int(row.get("rating",0)),
                "timestamp": row.get("timestamp","")
            }
        else:
            # 累加/更新 views rating
            question_map[qid]["views"] += int(row.get("views",0))
            oldr = question_map[qid]["rating"]
            newr = int(row.get("rating",0))
            if newr> oldr:
                question_map[qid]["rating"] = newr

    users = list(user_map.values())
    questions = list(question_map.values())
    return users, questions


def main():
    """
    主流程: 测试离线状态下的推荐。
      1) DataPipeline => rows
      2) aggregate => users, questions
      3) EnhancedContentAnalysis => question embedding + cluster
      4) EnhancedHotTopicAnalysis => hot_score
      5) UserProfileManager => 初始化画像(embedding,tag_weights可后续更新)
         并将 users 里的 rating => update_after_question
      6) 构建 RecommenderService => load question + hot
      7) 构建 CF => fit => load_cf_model
      8) user_id => recommend
    """
    ############################
    # 1) DataPipeline
    ############################
    input_csv = "data.csv"
    pipeline = DataPipeline(input_csv=input_csv)
    data_rows = pipeline.run_pipeline(drop_empty_description=False)
    if not data_rows:
        print("[main] No data from pipeline, abort.")
        return

    ############################
    # 2) aggregate => users, questions
    ############################
    users, questions = aggregate_users_and_questions(data_rows)
    print(f"[main] aggregated => users={len(users)}, questions={len(questions)}")

    ############################
    # 3) EnhancedContentAnalysis => embedding, cluster
    ############################
    analyzer = EnhancedContentAnalysis(
        use_bert=False,       # 若要用BERT => True, 需安装 transformers
        vector_size=64,
        n_clusters=10
    )
    analyzer.run_offline_content_analysis(questions)
    # 现在 question 里可能有 cluster_id, question["vector"]

    ############################
    # 4) EnhancedHotTopicAnalysis => hot_score
    ############################
    hot_analyzer = EnhancedHotTopicAnalysis(
        decay_factor=0.002,
        new_item_boost=3.0,
        new_item_days=10
    )
    hot_list = hot_analyzer.run_hot_analysis(questions, top_k=100)

    ############################
    # 5) UserProfile => 初始化 & update_after_question
    ############################
    profile_manager = UserProfileManager()
    # 由于 user_profile 里写好 "embedding":..., "tag_weights":...
    # 这里先仅进行 items => update_after_question
    # or 也可以把 "tag_weights" 逻辑再另行处理
    for u in users:
        uid = u["user_id"]
        # user_interest
        profile_manager.user_profiles[uid] = {
            "items": [],
            "embedding": np.array([], dtype=np.float32),
            "tag_weights": {},   # 这里若已有逻辑可写
            "done_questions": set()  # 也可放
        }
        # 记录 user_interest
        interest_str = u.get("user_interest","")
        if interest_str:
            # 可简单拆分或保留
            pass

        # update_after_question
        for q_item in u["questions"]:
            qid_str = str(q_item["question_id"])
            rating_val = int(q_item.get("rating",0))
            # find question
            qobj = next((qq for qq in questions if qq["id"]==qid_str), None)
            if not qobj:
                continue
            question_dict = {
                "id": qobj["id"],
                "tags": qobj["tags"],
                "rating": rating_val,
                "vector": qobj.get("vector", [])
            }
            # 这等同 profile_manager.update_after_question(uid, question_dict)
            # 这里为了演示
            if "done_questions" not in profile_manager.user_profiles[uid]:
                profile_manager.user_profiles[uid]["done_questions"]= set()
            profile_manager.user_profiles[uid]["done_questions"].add(qobj["id"])

    # PS: 若要用 item-based tag_weights => 需要写 update_after_question 细节

    ############################
    # 6) RecommenderService => load question + hot
    ############################
    recommender = RecommenderService(profile_manager)
    recommender.load_question_db(questions)
    recommender.update_hot_list(hot_list)

    ############################
    # 7) CF => fit => load_cf_model
    ############################
    interaction_rows = []
    for u in users:
        uid = u["user_id"]
        for q_item in u["questions"]:
            qid = str(q_item["question_id"])
            r = int(q_item.get("rating", 0))
            if r>0:
                interaction_rows.append( (uid, qid, r) )
    cf_model = CollaborativeFilterModel(sim_threshold=0.0)
    cf_model.fit(interaction_rows)
    recommender.load_cf_model(cf_model)

    ############################
    # 8) user_id => recommend
    ############################
    if len(sys.argv)>1:
        try:
            input_user_id = int(sys.argv[1])
        except:
            input_user_id = None
    else:
        try:
            input_user_id = int(input("请输入 user_id: ").strip())
        except:
            print("[main] invalid user_id, exit.")
            return

    if input_user_id not in profile_manager.user_profiles:
        print(f"[main] user_id={input_user_id} not in profiles.")
        return

    recs = recommender.recommend(input_user_id, top_n=5)
    if not recs:
        print(f"[main] user_id={input_user_id}, no rec found.")
    else:
        print(f"[main] user_id={input_user_id} => rec => {recs}")
        for i, qid in enumerate(recs, start=1):
            qinfo = recommender.question_db.get(qid, {})
            print(f" #{i} => QID={qid}, tags={qinfo.get('tags',[])}, "
                  f"rating={qinfo.get('rating',0)}, views={qinfo.get('views',0)}, "
                  f"hot_score={qinfo.get('hot_score',0)}")

if __name__=="__main__":
    main()
