# recommendation_system/offline_analysis.py

import math
import os
import csv
import re
import time
from datetime import datetime
from collections import defaultdict

import numpy as np

# ========== scikit-learn 用于 TF-IDF, PCA, KMeans等 ==========
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ========== transformers 用于 BERT embedding ==========
# 如果不用BERT可注释
try:
    from transformers import AutoTokenizer, AutoModel
    import torch

    BERT_AVAILABLE = True
except ImportError:
    print("[EnhancedContentAnalysis] transformers not installed. BERT embedding will be skipped.")
    BERT_AVAILABLE = False

# ========== Annoy 用于近似最近邻检索 (可选) ==========
try:
    from annoy import AnnoyIndex

    ANNOY_AVAILABLE = True
except ImportError:
    print("[EnhancedContentAnalysis] Annoy not installed. Vector index build will be skipped.")
    ANNOY_AVAILABLE = False


class EnhancedContentAnalysis:
    """
    增强版离线内容分析:
      - 提取多种特征(难度、知识点解析) => question["features"]
      - 支持TF-IDF+PCA, 也可选用BERT做文本向量
      - 支持KMeans做主题聚类 => question["cluster_id"]
      - 可选Annoy构建近似NN索引
    """

    def __init__(self,
                 use_bert=False,
                 bert_model_name="sentence-transformers/all-MiniLM-L6-v2",
                 use_annoy=True,
                 vector_size=128,
                 annoy_trees=10,
                 n_clusters=10):
        """
        :param use_bert: 是否启用BERT嵌入代替TF-IDF+PCA
        :param bert_model_name: huggingface上的bert模型
        :param use_annoy: 是否构建Annoy近似索引
        :param vector_size: 目标向量维度(若use_bert则=bert输出维度, 否则pca后大小)
        :param annoy_trees: Annoy构建树数
        :param n_clusters: KMeans聚类数
        """
        self.use_bert = use_bert and BERT_AVAILABLE
        self.bert_model_name = bert_model_name
        self.use_annoy = use_annoy and ANNOY_AVAILABLE
        self.vector_size = vector_size
        self.annoy_trees = annoy_trees
        self.n_clusters = n_clusters

        # scikit-learn相关
        self.vectorizer = None
        self.pca_model = None
        self.kmeans_model = None

        # BERT相关
        self.bert_tokenizer = None
        self.bert_model = None

        # 索引
        self.question_index_map = {}
        self.index_to_qid = []
        self.question_vectors = None
        self.annoy_index = None

    def init_bert(self):
        """
        如果 use_bert=True 则加载模型/Tokenizer
        """
        if not self.use_bert:
            return
        print(f"[EnhancedContentAnalysis] Loading BERT model {self.bert_model_name}...")
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
        self.bert_model = AutoModel.from_pretrained(self.bert_model_name)
        self.bert_model.eval()
        print("[EnhancedContentAnalysis] BERT loaded.")

    def build_question_features(self, questions):
        """
        示例: 分析题目文本, 提取“难度”“知识点”“额外特征”等, 存入 question["features"]
        这里仅演示一些简单规则:
          - 通过题目tags/描述中有 '字符串' => features["cat_string"]=1
          - or 用正则判断 'easy|hard' => features["diff"]=someValue
        你可自定义更多解析方式
        """
        for q in questions:
            q_text = q.get("text", "").lower()
            q_tags = q.get("tags", [])

            # 初始化 feature字典
            feat = {}

            # 1) 根据tags中是否含"字符串处理" "枚举算法"等
            if "字符串处理" in q_tags:
                feat["is_string"] = 1
            else:
                feat["is_string"] = 0

            if "枚举算法" in q_tags:
                feat["is_enum"] = 1
            else:
                feat["is_enum"] = 0

            # 2) 简单判断难度
            # e.g. 如果描述里含“简单|入门”，认为 easy=1
            if re.search(r"(简单|入门)", q_text):
                feat["difficulty"] = 1
            elif re.search(r"(复杂|困难)", q_text):
                feat["difficulty"] = 3
            else:
                feat["difficulty"] = 2  # 默认中等

            q["features"] = feat

    def get_text_for_embedding(self, q):
        """
        把题目文本 + tags + 额外特征拼装成embedding输入
        """
        text = q.get("text", "")
        tags = " ".join(q.get("tags", []))
        # 也可将q["features"]转换为字符串
        feat_str = ""
        if "features" in q:
            feat_items = [f"{k}={v}" for k, v in q["features"].items()]
            feat_str = " ".join(feat_items)
        combined = f"{text} {tags} {feat_str}"
        return combined.strip()

    def build_bert_vectors(self, questions):
        """
        对question使用BERT做句向量(简化: 取[CLS]token或mean pooling)
        """
        if not self.use_bert:
            return None

        print("[EnhancedContentAnalysis] Using BERT to generate embeddings...")
        vectors = []
        qid_list = []
        for q in questions:
            input_text = self.get_text_for_embedding(q)
            inputs = self.bert_tokenizer(input_text, return_tensors='pt', truncation=True, max_length=256)
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                # 取CLS向量(输出[0][:,0,:]) or mean pooling
                cls_vec = outputs.last_hidden_state[:, 0, :]
                # 也可以mean pooling => outputs.last_hidden_state.mean(dim=1)
                embedding = cls_vec[0].numpy()  # shape: [768], e.g.
            vectors.append(embedding)
            qid_list.append(q["id"])

        vectors = np.array(vectors, dtype=np.float32)
        return vectors, qid_list

    def build_tfidf_pca_vectors(self, questions):
        """
        传统TF-IDF + PCA
        """
        # 1) 准备文本
        corpus = []
        qid_list = []
        for q in questions:
            combined_text = self.get_text_for_embedding(q)
            corpus.append(combined_text)
            qid_list.append(q["id"])

        if not corpus:
            return None, None

        # 2) TF-IDF
        print(f"[EnhancedContentAnalysis] Building TF-IDF for {len(corpus)} items.")
        self.vectorizer = TfidfVectorizer(max_features=30000, stop_words='english')
        tfidf_mat = self.vectorizer.fit_transform(corpus)  # shape [N, vocab_size]

        dense_mat = tfidf_mat.toarray()
        vsize = dense_mat.shape[1]
        if vsize > self.vector_size:
            print("[EnhancedContentAnalysis] Running PCA to reduce dimension.")
            self.pca_model = PCA(n_components=self.vector_size)
            reduced = self.pca_model.fit_transform(dense_mat)
            vectors = reduced.astype(np.float32)
        else:
            # 仅padding
            vectors = []
            for row in dense_mat:
                pad_len = self.vector_size - len(row)
                if pad_len > 0:
                    padded = np.hstack([row, np.zeros(pad_len, dtype=np.float32)])
                else:
                    padded = row[:self.vector_size]
                vectors.append(padded)
            vectors = np.array(vectors, dtype=np.float32)

        return vectors, qid_list

    def build_vectors_and_clusters(self, questions):
        """
        主流程: build question["features"] => embedding => cluster => set question["cluster_id"]
        """
        # 1) 提取额外特征
        self.build_question_features(questions)

        # 2) embedding
        if self.use_bert:
            # 加载bert
            self.init_bert()
            vectors, qid_list = self.build_bert_vectors(questions)
            if vectors is None:
                return
            self.vector_size = vectors.shape[1]
        else:
            # tfidf + pca
            vectors, qid_list = self.build_tfidf_pca_vectors(questions)
            if vectors is None:
                return

        self.question_vectors = vectors
        self.index_to_qid = qid_list
        self.question_index_map = {qid_list[i]: i for i in range(len(qid_list))}

        # 3) KMeans聚类
        print(f"[EnhancedContentAnalysis] KMeans n_clusters={self.n_clusters}")
        self.kmeans_model = KMeans(n_clusters=self.n_clusters, random_state=42)
        cluster_labels = self.kmeans_model.fit_predict(vectors)
        # 将 cluster_id 写回 question
        for i, cid in enumerate(cluster_labels):
            qid = qid_list[i]
            # 找到对应question
            # (如果数量大,可建dict.这里示例)
            # 也可以 question[i]["cluster_id"]=cid
            # 这里再搜索
            # ...
            # 简洁: questions[i]["cluster_id"] = cid  (前提: i对应question列表顺序)
            # 但要保证 i和question[i]能对应
            # 这里演示: 先构建qid-> cluster
        qid_to_cluster = {}
        for i, cid in enumerate(cluster_labels):
            qid_to_cluster[qid_list[i]] = cid

        for q in questions:
            qid = q["id"]
            if qid in qid_to_cluster:
                q["cluster_id"] = int(qid_to_cluster[qid])

    def build_vector_index(self):
        """
        构建Annoy索引
        """
        if not self.use_annoy:
            print("[EnhancedContentAnalysis] Annoy not used or missing.")
            return
        if self.question_vectors is None:
            print("[EnhancedContentAnalysis] question_vectors is None, skip annoy.")
            return
        n = len(self.question_vectors)
        self.annoy_index = AnnoyIndex(self.vector_size, metric='angular')
        print(f"[EnhancedContentAnalysis] Building Annoy index, n={n}, metric=angular")
        for i in range(n):
            self.annoy_index.add_item(i, self.question_vectors[i].tolist())
        self.annoy_index.build(self.annoy_trees)
        print(f"[EnhancedContentAnalysis] Annoy index built with {n} items, trees={self.annoy_trees}.")

    def run_offline_content_analysis(self, questions):
        """
        全流程:
          1) 构建题目多维度特征
          2) 生成embedding(bert or tfidf+pca)
          3) KMeans cluster => question["cluster_id"]
          4) 构建Annoy(可选)
        """
        if not questions:
            print("[EnhancedContentAnalysis] No questions to analyze.")
            return
        self.build_vectors_and_clusters(questions)
        self.build_vector_index()
        print("[EnhancedContentAnalysis] Enhanced content analysis done.")


class EnhancedHotTopicAnalysis:
    """
    带有“时间衰减”和“新题boost”的热门度计算
    """

    def __init__(self, decay_factor=0.001, new_item_boost=5.0, new_item_days=7):
        """
        :param decay_factor: 时间衰减系数(越大,越快衰减)
        :param new_item_boost: 新题额外加成(在7天内)
        :param new_item_days: 多少天内视为新题
        """
        self.decay_factor = decay_factor
        self.new_item_boost = new_item_boost
        self.new_item_days = new_item_days

    def parse_timestamp(self, ts_str):
        # 假设timestamp是 yyyy-MM-ddTHH:mm:ss
        # 这里简单实现, 如果解析失败则返回None
        try:
            return datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S")
        except:
            return None

    def compute_hot_score(self, question):
        """
        1) base_score = 0.6 * rating + 0.4 * views
        2) time_decay => factor = exp(-decay_factor * day_diff)
        3) new_item => if day_diff < new_item_days => score += new_item_boost
        """
        rating = question.get("rating", 0)
        views = question.get("views", 0)
        base_score = 0.6 * rating + 0.4 * views

        # parse timestamp if any
        ts_str = question.get("timestamp", "")
        t_post = self.parse_timestamp(ts_str)
        day_diff = 9999
        if t_post:
            day_diff = (datetime.now() - t_post).days
        # time decay
        factor = math.exp(-self.decay_factor * day_diff)
        final_score = base_score * factor
        # new item boost
        if day_diff <= self.new_item_days:
            final_score += self.new_item_boost

        return final_score

    def generate_hot_list(self, questions, top_k=100):
        if not questions:
            return []
        for q in questions:
            sc = self.compute_hot_score(q)
            q["hot_score"] = sc
        sorted_q = sorted(questions, key=lambda x: x["hot_score"], reverse=True)
        return sorted_q[:top_k]

    def run_hot_analysis(self, questions, top_k=100):
        hot_list = self.generate_hot_list(questions, top_k=top_k)
        print(f"[EnhancedHotTopicAnalysis] Generated top {len(hot_list)} hot items with decay/new-boost.")
        return hot_list


def advanced_offline_demo(questions_csv="data.csv"):
    """
    演示:
      - 读取 question数据
      - EnhancedContentAnalysis => embedding + cluster
      - EnhancedHotTopicAnalysis => time decay & new item boost
    """
    # 1) 读取 CSV -> 构造 question list
    from data_pipeline import DataPipeline
    pipeline = DataPipeline(input_csv=questions_csv)
    data_rows = pipeline.run_pipeline(drop_empty_description=False)
    # 这里把 question去重
    question_map = {}
    for row in data_rows:
        qid = str(row["question_id"])
        if not qid:
            continue
        if qid not in question_map:
            # build
            qinfo = {
                "id": qid,
                "text": row.get("question_description", ""),
                "tags": row.get("question_keywords", "").split(),
                "views": int(row.get("views", 0)),
                "rating": int(row.get("rating", 0)),
                "timestamp": row.get("timestamp", "")
            }
            question_map[qid] = qinfo
        else:
            # 累计 views
            question_map[qid]["views"] += int(row.get("views", 0))
            oldR = question_map[qid]["rating"]
            newR = int(row.get("rating", 0))
            if newR > oldR:
                question_map[qid]["rating"] = newR

    questions = list(question_map.values())
    print(f"[advanced_offline_demo] Collected {len(questions)} unique questions from CSV.")

    # 2) EnhancedContentAnalysis
    analyzer = EnhancedContentAnalysis(use_bert=False,  # 先关掉BERT,若要用 => True
                                       vector_size=64,
                                       n_clusters=10)
    analyzer.run_offline_content_analysis(questions)
    # 现在 question里会新增 cluster_id, features, vector?

    # 3) EnhancedHotTopicAnalysis
    hot_analyzer = EnhancedHotTopicAnalysis(decay_factor=0.002,
                                            new_item_boost=3.0,
                                            new_item_days=10)
    hot_list = hot_analyzer.run_hot_analysis(questions, top_k=10)

    print("\n[advanced_offline_demo] ***Top10 hot questions***")
    for i, q in enumerate(hot_list, start=1):
        print(f" #{i} => QID={q['id']}, rating={q['rating']}, views={q['views']}, "
              f"hot_score={q['hot_score']:.2f}, cluster_id={q.get('cluster_id', '?')}")

    # 4) (可选) 你可以将 question更新后的信息(embedding, cluster_id, hot_score)
    #    存储到本地json/csv, 以便后续在线加载

    print("[advanced_offline_demo] done.")


if __name__ == "__main__":
    advanced_offline_demo("data.csv")
