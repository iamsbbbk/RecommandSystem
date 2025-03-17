# models/test.py
import csv

def filter_csv_by_description(input_csv, output_csv):
    """
    从 input_csv 读取所有行, 只保留 question_description 不为空的行,
    然后将结果写入 output_csv.
    """
    with open(input_csv, 'r', encoding='utf-8') as fin, \
         open(output_csv, 'w', newline='', encoding='utf-8') as fout:

        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames  # 保留原有列名
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            description = row.get("question_description", "").strip()
            # 如果 description 不为空, 则保留
            if description:
                writer.writerow(row)

    print(f"[filter_csv_by_description] Done. Filtered CSV => {output_csv}")


def main():
    input_file = "consolidated_data.csv"    # 你的原合并后CSV
    output_file = "data.csv"  # 过滤后生成的新CSV

    filter_csv_by_description(input_file, output_file)


if __name__ == "__main__":
    main()
