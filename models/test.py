#!/usr/bin/env python3
# merge_user_problem.py

import os
import json
import csv
from typing import List


def build_question_description(problem_data: dict) -> str:
    """
    将题目中的 "说明"、"输入格式"、"输出格式"、"样例" 等合并为一个大文本，
    并在各部分之间加入换行，便于后续处理和显示。
    """
    parts = []
    if "说明" in problem_data and problem_data["说明"]:
        parts.append(problem_data["说明"])
    if "输入格式" in problem_data and problem_data["输入格式"]:
        parts.append("输入格式: " + str(problem_data["输入格式"]))
    if "输出格式" in problem_data and problem_data["输出格式"]:
        parts.append("输出格式: " + str(problem_data["输出格式"]))
    if "样例" in problem_data and problem_data["样例"]:
        sample_part = "样例:"
        if isinstance(problem_data["样例"], dict):
            sample_input = problem_data["样例"].get("输入", "")
            sample_output = problem_data["样例"].get("输出", "")
            sample_part += f"\n输入: {sample_input}\n输出: {sample_output}"
        else:
            sample_part += str(problem_data["样例"])
        parts.append(sample_part)
    return "\n".join(parts)


def load_problem_info(problem_folder: str, qid: str) -> (str, str):
    """
    根据题目id (qid) 读取题目文件夹中对应的 problem.json，
    提取出题目的关键词和题目描述信息。
    如果文件不存在或读取失败，则返回 ("", "")。
    """
    sub_path = os.path.join(problem_folder, qid)
    problem_json_path = os.path.join(sub_path, "problem.json")
    if not os.path.exists(problem_json_path):
        return "", ""
    try:
        with open(problem_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"读取 {problem_json_path} 失败: {e}")
        return "", ""
    keywords_list = data.get("关键词", [])
    if isinstance(keywords_list, str):
        keywords_list = [kw.strip() for kw in keywords_list.split() if kw.strip()]
    question_keywords = " ".join(keywords_list)
    question_description = build_question_description(data)
    return question_keywords, question_description


def load_user_json(user_json_path: str) -> dict:
    """
    读取单个用户JSON文件，返回解析后的字典。
    如果读取失败，返回空字典。
    """
    if not os.path.exists(user_json_path):
        return {}
    try:
        with open(user_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"读取 {user_json_path} 失败: {e}")
        return {}


def main():
    # 直接在代码中设置各个文件夹的路径
    user_folder = "../user"  # 用户JSON文件所在文件夹路径（请替换为实际路径）
    problem_folder = "../JSON_FD_2/JSON_FD_2"  # 题目文件夹所在路径（请替换为实际路径）
    output_csv = "data.csv"  # 输出CSV文件名

    fieldnames = [
        "user_id",
        "question_id",
        "timestamp",
        "views",
        "rating",
        "user_interest",
        "question_keywords",
        "question_description"
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        # 遍历用户文件夹下的所有JSON文件
        user_files = [f for f in os.listdir(user_folder) if f.endswith(".json")]
        if not user_files:
            print("用户文件夹中未找到任何JSON文件。")
            return

        for user_file in user_files:
            user_json_path = os.path.join(user_folder, user_file)
            user_data = load_user_json(user_json_path)
            if not user_data:
                continue

            user_id = user_data.get("user_id", 0)
            interest_list = user_data.get("interest", [])
            if isinstance(interest_list, list):
                user_interest = " ".join(str(i) for i in interest_list)
            else:
                user_interest = str(interest_list)

            questions = user_data.get("questions", [])
            for q in questions:
                qid = str(q.get("question_id", ""))
                timestamp = q.get("timestamp", "")
                views = q.get("views", 0)
                rating = q.get("rating", 0)

                question_keywords, question_description = load_problem_info(problem_folder, qid)
                row = {
                    "user_id": user_id,
                    "question_id": qid,
                    "timestamp": timestamp,
                    "views": views,
                    "rating": rating,
                    "user_interest": user_interest,
                    "question_keywords": question_keywords,
                    "question_description": question_description
                }
                writer.writerow(row)

    print(f"已完成合并，结果保存在 {output_csv}。")


if __name__ == "__main__":
    main()
