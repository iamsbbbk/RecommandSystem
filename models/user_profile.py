# models/user_profile.py

import os
import csv
import statistics
import numpy as np
from typing import Dict, List, Set

class UserProfileManager:


    def __init__(self):

        self.user_profiles: Dict[int, Dict] = {}

        self.weight_rating = 0.5
        self.weight_fav = 0.3
        self.weight_correct = 0.2

    def load_user_interactions(self, csv_path: str):

        if not os.path.exists(csv_path):
            print(f"[UserProfileManager] CSV {csv_path} not found.")
            return

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    uid = int(row.get("user_id","0"))
                    qid = row.get("question_id","").strip()
                except:
                    continue
                if not qid:
                    continue
                r = float(row.get("rating", "0"))
                fav = float(row.get("is_fav", "0"))
                corr = float(row.get("correct_rate","0"))

                raw_score = (self.weight_rating*r
                             + self.weight_fav*fav
                             + self.weight_correct*corr)

                if uid not in self.user_profiles:
                    self.user_profiles[uid] = {
                        "items": [],
                        "embedding": np.array([], dtype=np.float32),
                        "tag_weights": {}
                    }
                self.user_profiles[uid]["items"].append({
                    "question_id": qid,
                    "raw_score": raw_score
                })

    def normalize_scores_zscore(self):

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

        if user_id not in self.user_profiles:
            return []
        items = self.user_profiles[user_id]["items"]
        return [(x["question_id"], x.get("normalized_score", 0.0))
                for x in items]

    def list_all_users(self):
        return list(self.user_profiles.keys())

    def get_user_profile(self, user_id: int) -> Dict:

        if user_id not in self.user_profiles:
            return None
        return self.user_profiles[user_id]


def demo_profile_enhancement():

    upm = UserProfileManager()

    upm.load_user_interactions("data.csv")
    upm.normalize_scores_zscore()


    all_users = upm.list_all_users()
    if not all_users:
        return
    sample_uid = all_users[0]
    items = upm.get_user_item_scores(sample_uid)
    print(f"[demo] user_id={sample_uid}, item_scores => {items[:10]}")

    prof = upm.get_user_profile(sample_uid)
    print(f"[demo] user_id={sample_uid}, embedding =>", prof["embedding"])
    print(f"[demo] user_id={sample_uid}, tag_weights =>", prof["tag_weights"])


if __name__=="__main__":
    demo_profile_enhancement()
