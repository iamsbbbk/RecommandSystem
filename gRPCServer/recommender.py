# gRPCServer/recommender.py

class RecommenderService:
    """
    演示用的业务逻辑类。
    在真实项目中，你会存储用户交互信息到数据库/内存，更新协同过滤模型或用户画像等。
    """

    def __init__(self):
        # 用一个简单的 in-memory 结构记录 [user_id -> list of interactions]
        self.user_data = {}
        # 题库/推荐数据的简单模拟(实际可能来自数据库)
        self.question_db = {
            "1001": {"tags": ["dp","graph"], "rating": 4.0, "views": 120, "hot_score": 3.5},
            "1002": {"tags": ["search","greedy"], "rating": 3.5, "views": 200, "hot_score": 5.1},
            "496":  {"tags": ["枚举算法"], "rating": 4.5, "views": 88,  "hot_score": 2.2},
            # ... 更多题
        }

    def update_interactions(self, user_id, interaction):
        """
        更新用户交互, interaction是一个 dict，如:
          {
            'question_id': '496',
            'rating': 4,
            'views': 1,
            'timestamp': '2023-09-25T10:20:00',
            ...
          }
        """
        if user_id not in self.user_data:
            self.user_data[user_id] = []
        self.user_data[user_id].append(interaction)

    def recommend(self, user_id, top_n=5, question_id=None, rating=0.0, views=0):
        """
        返回一个最简单的推荐列表(模拟).
        实际上你可使用 CF、标签召回等。
        """
        # 如果 user_id 不在数据里, 说明没多少信息 => 返回 question_db 中 random topN
        if user_id not in self.user_data:
            return list(self.question_db.keys())[:top_n]

        # 示例策略: 如果 rating>3 => 认为用户喜欢 question_id => 找 tag相近的题
        rec_candidates = list(self.question_db.keys())
        if question_id and rating>3:
            # 仅演示: 如果 rating>3 => 优先过滤出 tags 与 question_id相似
            qtags = self.question_db.get(question_id, {}).get("tags", [])
            # 找与qtags有交集的candidate
            rec_candidates = [qid for qid in self.question_db if any(t in self.question_db[qid]["tags"] for t in qtags)]

        # 截取topN
        return rec_candidates[:top_n]
