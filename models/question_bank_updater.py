# models/question_bank_updater.py

import os
import csv
import json
import time
from datetime import datetime


class QuestionBankUpdater:
    """
    每5分钟更新题库列表，将题目数据从 CSV 文件中聚合后保存为 JSON 文件。
    """

    def __init__(self, input_csv="data.csv", output_json="question_bank.json"):
        self.input_csv = input_csv
        self.output_json = output_json

    def load_data(self):
        if not os.path.exists(self.input_csv):
            print(f"[QuestionBankUpdater] Input CSV '{self.input_csv}' not found!")
            return []
        data_rows = []
        try:
            with open(self.input_csv, "r", encoding="utf-8-sig") as f:
                # 读取文件前1024字节，检测分隔符
                sample = f.read(1024)
                f.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(sample)
                    print(f"[QuestionBankUpdater] Detected delimiter: '{dialect.delimiter}'")
                except csv.Error:
                    dialect = csv.excel
                    dialect.delimiter = "\t"
                    print(f"[QuestionBankUpdater] Using default delimiter: '\\t'")

                # 读取 header 行，判断是否包含逗号
                first_line = f.readline()
                if "," in first_line:
                    # 如果 header 中包含逗号，则认为文件是逗号分隔
                    dialect.delimiter = ","
                    print("[QuestionBankUpdater] Overriding dialect delimiter to comma based on header line.")
                    f.seek(0)
                else:
                    f.seek(0)

                reader = csv.DictReader(f, dialect=dialect)

                # 如果 header 只有一个字段且包含逗号，则手动拆分
                if reader.fieldnames and len(reader.fieldnames) == 1 and ',' in reader.fieldnames[0]:
                    original_header = reader.fieldnames[0]
                    new_header = [col.strip() for col in original_header.split(',')]
                    reader.fieldnames = new_header
                    print(f"[QuestionBankUpdater] Overriding header fields to: {reader.fieldnames}")
                else:
                    print(f"[QuestionBankUpdater] Detected header fields: {reader.fieldnames}")

                for row in reader:
                    # 将每个字段转换为字符串，处理 None 和列表情况
                    for key, field in row.items():
                        if field is None:
                            row[key] = ""
                        elif isinstance(field, list):
                            row[key] = " ".join(field)
                        else:
                            row[key] = str(field)
                    # 仅保留 question_id 非空的行
                    if (row.get("question_id") or "").strip():
                        data_rows.append(row)
            print(f"[QuestionBankUpdater] Loaded {len(data_rows)} rows from {self.input_csv}")
        except Exception as e:
            print(f"[QuestionBankUpdater] Error reading CSV: {e}")
        return data_rows

    def aggregate_questions(self, data_rows):
        """
        聚合题目信息，返回题目列表，每个题目为一个字典，包含：
          - id, text, tags, views, rating, timestamp, hot_score
        过滤条件：
          1. question_id 必须能转换为正整数；
          2. question_description 长度不少于 20 个字符。
        """
        question_map = {}
        for row in data_rows:
            qid_str = (row.get("question_id") or "").strip()
            if not qid_str:
                continue
            try:
                qid_int = int(qid_str)
                if qid_int <= 0:
                    continue
            except ValueError:
                continue

            desc = (row.get("question_description") or "").strip()
            if len(desc) < 20:
                continue

            if qid_str not in question_map:
                kw_str = (row.get("question_keywords") or "").strip()
                tags = kw_str.split() if kw_str else []
                try:
                    views = int(row.get("views", 0))
                except:
                    views = 0
                try:
                    rating = int(row.get("rating", 0))
                except:
                    rating = 0
                timestamp = (row.get("timestamp") or "").strip()
                hot_score = 0.6 * rating + 0.4 * views
                question_map[qid_str] = {
                    "id": qid_str,
                    "text": desc,
                    "tags": tags,
                    "views": views,
                    "rating": rating,
                    "timestamp": timestamp,
                    "hot_score": hot_score
                }
            else:
                try:
                    additional_views = int(row.get("views", 0))
                except:
                    additional_views = 0
                question_map[qid_str]["views"] += additional_views
                try:
                    rating = int(row.get("rating", 0))
                except:
                    rating = 0
                if rating > question_map[qid_str]["rating"]:
                    question_map[qid_str]["rating"] = rating
                question_map[qid_str]["hot_score"] = 0.6 * question_map[qid_str]["rating"] + 0.4 * \
                                                     question_map[qid_str]["views"]
        return list(question_map.values())

    def update_question_bank(self):
        data_rows = self.load_data()
        if not data_rows:
            print("[QuestionBankUpdater] No data loaded. Check CSV file and delimiter settings.")
            return
        question_list = self.aggregate_questions(data_rows)
        try:
            with open(self.output_json, "w", encoding="utf-8") as f:
                json.dump(question_list, f, ensure_ascii=False, indent=4)
            print(
                f"[QuestionBankUpdater] Updated question bank with {len(question_list)} questions. Saved to {self.output_json}")
        except Exception as e:
            print(f"[QuestionBankUpdater] Error writing JSON: {e}")

    def run(self):
        """
        每5分钟更新一次题库
        """
        while True:
            print(f"[QuestionBankUpdater] Updating question bank at {datetime.now()}")
            self.update_question_bank()
            time.sleep(300)  # 300秒 = 5分钟


if __name__ == "__main__":
    updater = QuestionBankUpdater()
    updater.run()
