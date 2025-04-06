# recommendation_system/simulate_online_recommendation.py

import random
import numpy as np
import time

from data_pipeline import DataPipeline
from user_profile import UserProfileManager
from collaborative_filter import CollaborativeFilterModel
from recommender import RecommenderService
from offline_analysis import EnhancedContentAnalysis, EnhancedHotTopicAnalysis

def simulate_online_usage():
    pipeline = DataPipeline("data.csv")
    data_rows = pipeline.run_pipeline(drop_empty_description=False)
    if not data_rows:
        print("[simulate_online] no data, abort.")
        return

    question_map = {}
    for row in data_rows:
        qid = str(row["question_id"])
        if not qid:
            continue
        if qid not in question_map:
            question_map[qid] = {
                "id": qid,
                "text": row.get("question_description",""),
                "tags": row.get("question_keywords","").split(),
                "views": int(row.get("views",0)),
                "rating": int(row.get("rating",0)),
                "timestamp": row.get("timestamp","")
            }
        else:
            question_map[qid]["views"] += int(row.get("views",0))
            oldr = question_map[qid]["rating"]
            newr = int(row.get("rating",0))
            if newr>oldr:
                question_map[qid]["rating"] = newr

    questions = list(question_map.values())

    # =========== offline steps =============
    analyzer = EnhancedContentAnalysis(use_bert=False, vector_size=64, n_clusters=10)
    analyzer.run_offline_content_analysis(questions)

    hot_analyzer = EnhancedHotTopicAnalysis(decay_factor=0.002, new_item_boost=3.0, new_item_days=10)
    hot_list = hot_analyzer.run_hot_analysis(questions, top_k=100)

    up_manager = UserProfileManager()
    # Recommender
    recommender = RecommenderService(up_manager)
    recommender.load_question_db(questions)
    recommender.update_hot_list(hot_list)
    cf_model = CollaborativeFilterModel(sim_threshold=0.0)
    cf_model.fit([]) # init empty
    recommender.load_cf_model(cf_model)

    # create some user
    user_ids = [101,102,103]
    for uid in user_ids:
        up_manager.user_profiles[uid] = {
            "items": [],
            "embedding": np.array([], dtype=np.float32),
            "tag_weights": {},
            "done_questions": set(),
            "recently_recommended": set()
        }

    rounds = 10
    all_data=[]
    for i in range(rounds):
        print(f"\n=== Round {i+1} ===")
        user_id = random.choice(user_ids)
        recs = recommender.recommend(user_id, top_n=5)
        if not recs:
            print(f"[simulate_online] user={user_id} => no rec!")
            continue
        print(f"[simulate_online] user={user_id} => rec => {recs}")

        # user clicks 1~2
        clicked = random.sample(recs, k=random.randint(1,2))
        print(f"[simulate_online] user={user_id} clicked => {clicked}")

        # add interactions => rating ~ 3..5
        new_interactions = []
        for it in clicked:
            rating_val = random.randint(3,5)
            new_interactions.append( (user_id,it,rating_val) )
            # update user done
            up_manager.user_profiles[user_id]["done_questions"].add(it)

        # re-fit CF with old + new
        old_data=[]
        for u,items_map in cf_model.user_items.items():
            for i_2, rat in items_map.items():
                old_data.append((u,i_2,rat))
        merged = old_data + new_interactions
        cf_model.fit(merged)
        recommender.load_cf_model(cf_model)

        # optional: prune 'recently_recommended' if too big
        if len(up_manager.user_profiles[user_id]["recently_recommended"])>20:
            # remove older 5
            arr = list(up_manager.user_profiles[user_id]["recently_recommended"])
            remove_ = random.sample(arr,k=5)
            for x in remove_:
                up_manager.user_profiles[user_id]["recently_recommended"].discard(x)

        time.sleep(1)

    print("\n[simulate_online] done.")


if __name__=="__main__":
    simulate_online_usage()
