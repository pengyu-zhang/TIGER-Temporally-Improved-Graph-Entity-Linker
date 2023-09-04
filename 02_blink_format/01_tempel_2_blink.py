import jsonlines
import json
import time
from collections import defaultdict
from typing import Dict


class FileProcessor:
    def __init__(self, input_base_dir: str, output_base_dir: str, directories: list, years: range):
        self.input_base_dir = input_base_dir
        self.output_base_dir = output_base_dir
        self.directories = directories
        self.years = years
        self.qid_mention_count = defaultdict(int)

    def process_files(self):
        start_time = time.time()
        print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

        for directory in self.directories:
            for year in self.years:
                input_file = f"{self.input_base_dir}/{directory}/{year}-01-01T00_00_00Z_{directory}.jsonl"
                output_file = f"{self.output_base_dir}/{year}/{year}_{directory}.jsonl"
                self._process_single_file(input_file, output_file, f"{year}_{directory}")

        end_time = time.time()
        print(f"Completed at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
        print(f"Time cost: {(end_time - start_time) / 60:.2f} minutes\n")

    def _process_single_file(self, input_file: str, output_file: str, world: str) -> None:
        with jsonlines.open(input_file) as f, open(output_file, "a", encoding="utf-8") as output_f:
            for line in f:
                new_dict = {
                    "context_left": line["context_left_bert"],
                    "context_right": line["context_right_bert"],
                    "mention": line["mention"],
                    "label": line["target_bert"],
                    "label_id": line["target_page_id"],
                    "label_title": line["target_title"],
                    "world": world,
                    "qid": line["target_qid"],
                    "category": line["category"],
                    "mention_id": self._get_mention_id(line["target_qid"]),
                }

                json_str = json.dumps(new_dict, ensure_ascii=False)
                output_f.write(json_str + "\n")

    def _get_mention_id(self, qid: str) -> str:
        self.qid_mention_count[qid] += 1
        return f"qid_{qid}_mention_{self.qid_mention_count[qid]}"


if __name__ == "__main__":
    input_dir = "D:/datasets/06_dataset_7_4/tempel_v1.0_only_bert_tokenized"
    output_dir = "D:/datasets/06_dataset_7_4/03_blink_format"
    dirs = ["train", "validation", "test"]
    years_range = range(2013, 2023)

    processor = FileProcessor(input_dir, output_dir, dirs, years_range)
    processor.process_files()
