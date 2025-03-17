# models/data_pipline.py
import os
import csv
import json
from typing import List, Dict, Optional


class DataPipeline:
    """
    主要功能：
      - 读取 CSV
      - 清洗转换
      - (可选) 过滤空描述/空兴趣
      - 返回（或保存）结构化结果
    """

    def __init__(self, input_csv: str, output_csv: Optional[str] = None):
        """
        :param input_csv: 例如 "data.csv" 的路径
        :param output_csv: 如果希望将清洗后的结果写到新的 CSV，可指定文件名；若为 None，则只返回数据而不写文件
        """
        self.input_csv = input_csv
        self.output_csv = output_csv if output_csv else None

    def load_data(self) -> List[Dict]:
        """
        读取 input_csv 并将其解析为 list of dict。
        :return: [{'user_id':..., 'question_id':..., ...}, ...]
        """
        if not os.path.exists(self.input_csv):
            print(f"[DataPipeline] input CSV '{self.input_csv}' not found!")
            return []

        data_rows = []
        try:
            with open(self.input_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    data_rows.append(row)
            print(f"[DataPipeline] Loaded {len(data_rows)} rows from {self.input_csv}")
        except Exception as e:
            print(f"[DataPipeline] Error reading CSV: {e}")
            return []

        return data_rows

    def clean_data(self, data_rows: List[Dict], drop_empty_description: bool = False) -> List[Dict]:
        """
        对读入的数据进行清洗和转换：
         - views, rating 转为 int
         - user_interest, question_keywords 可根据需求拆分成 list 或保留原字符串
         - 如需要，可以过滤 question_description 为空的行
        """
        cleaned = []
        for row in data_rows:
            # 1) 强制转换 user_id, question_id, views, rating
            #    假设 user_id, question_id 是 int；若 CSV 中可能是 str，需要转一下
            #    如果 CSV 中是 "1"，则 int("1") => 1
            #    出现异常时可以继续或跳过
            try:
                user_id = int(row.get("user_id", 0))
            except ValueError:
                user_id = 0
            try:
                question_id = int(row.get("question_id", 0))
            except ValueError:
                question_id = 0
            try:
                views = int(row.get("views", 0))
            except ValueError:
                views = 0
            try:
                rating = int(row.get("rating", 0))
            except ValueError:
                rating = 0

            # 2) 解析其他列：timestamp, user_interest, question_keywords, question_description
            timestamp = row.get("timestamp", "").strip()

            user_interest_str = row.get("user_interest", "").strip()
            # 如果需要将其变为 list，可以在这里分割（如空格/逗号）
            # 如果只想保留原字符串，也可以直接用
            # 例如： interest_list = user_interest_str.split() if user_interest_str else []
            # 这里演示保留原字符串
            user_interest = user_interest_str

            question_keywords_str = row.get("question_keywords", "").strip()
            # 同样也可 split
            question_keywords = question_keywords_str

            question_desc = row.get("question_description", "").strip()

            # 3) 如果需要过滤空描述
            if drop_empty_description and not question_desc:
                # 跳过
                continue

            new_item = {
                "user_id": user_id,
                "question_id": question_id,
                "timestamp": timestamp,
                "views": views,
                "rating": rating,
                "user_interest": user_interest,      # or interest_list
                "question_keywords": question_keywords,   # or keywords_list
                "question_description": question_desc
            }
            cleaned.append(new_item)

        print(f"[DataPipeline] Cleaned rows: {len(cleaned)}")
        return cleaned

    def save_data(self, cleaned_rows: List[Dict]):
        """
        如果指定了 output_csv，则将清洗后的数据写入新的 CSV 文件。
        """
        if not self.output_csv:
            # 未指定输出文件，则不写
            print("[DataPipeline] No output_csv specified. Skipping save.")
            return

        if not cleaned_rows:
            print("[DataPipeline] Nothing to save, cleaned_rows is empty.")
            return

        # 取所有字段名
        fieldnames = list(cleaned_rows[0].keys())

        try:
            with open(self.output_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in cleaned_rows:
                    writer.writerow(row)
            print(f"[DataPipeline] Saved {len(cleaned_rows)} rows to {self.output_csv}")
        except Exception as e:
            print(f"[DataPipeline] Error saving CSV: {e}")

    def run_pipeline(self, drop_empty_description: bool = False) -> List[Dict]:
        """
        主流程：读取 -> 清洗 -> (可选)过滤空描述 -> 写出 & 返回
        :param drop_empty_description: 若为 True，则丢掉 question_description 为空的行
        :return: 清洗后的列表
        """
        print("[DataPipeline] Starting pipeline with data.csv =>", self.input_csv)
        rows = self.load_data()
        cleaned = self.clean_data(rows, drop_empty_description=drop_empty_description)
        self.save_data(cleaned)
        print("[DataPipeline] Pipeline completed.")
        return cleaned


def demo_data_pipeline():
    """
    演示用法
    假设 data.csv 就在当前目录
    """
    input_csv = "data.csv"          # 你的 CSV 路径
    output_csv = "cleaned_data.csv" # 输出文件

    pipeline = DataPipeline(input_csv, output_csv)

    # 例如想过滤掉 question_description 为空的行
    final_data = pipeline.run_pipeline(drop_empty_description=True)

    # 后续你可在内存中使用 final_data 训练模型，或加载 cleaned_data.csv
    print("\n[demo_data_pipeline] Sample row =>", final_data[0] if final_data else None)


if __name__ == "__main__":
    demo_data_pipeline()
