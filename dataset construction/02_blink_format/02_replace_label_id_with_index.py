import jsonlines
import json
import time
from typing import Dict


class FileProcessor:
    def __init__(self, input_base_dir: str, output_base_dir: str, directories: list, years: range):
        self.input_base_dir = input_base_dir
        self.output_base_dir = output_base_dir
        self.directories = directories
        self.years = years

    def process_files(self):
        start_time = time.time()
        print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

        for directory in self.directories:
            for year in self.years:
                document_file = f"{self.input_base_dir}/02_documents_no_duplicates/{year}_{directory}.json"
                input_file = f"{self.input_base_dir}/03_blink_format/{year}/{year}_{directory}.jsonl"
                output_file = f"{self.output_base_dir}/{year}/{year}_{directory}.jsonl"
                self._process_single_file(document_file, input_file, output_file)

        end_time = time.time()
        print(f"Completed at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
        print(f"Time cost: {(end_time - start_time) / 60:.2f} minutes\n")

    def _process_single_file(self, document_file: str, input_file: str, output_file: str) -> None:
        document_ids = self._get_document_ids(document_file)

        with jsonlines.open(input_file) as f_in, open(output_file, "w", encoding="utf-8") as f_out:
            for line in f_in:
                label_id = str(line["label_id"])
                if label_id in document_ids:
                    line["label_id"] = document_ids[label_id]
                    f_out.write(json.dumps(line, ensure_ascii=False) + '\n')

    def _get_document_ids(self, document_file: str) -> Dict[str, int]:
        document_ids = {}
        with jsonlines.open(document_file, "r") as f:
            for i, line in enumerate(f):
                document_ids[str(line["document_id"])] = i
        return document_ids


if __name__ == "__main__":
    input_dir = "D:/datasets/06_dataset_7_4"
    output_dir = "D:/datasets/06_dataset_7_4/04_blink_format_with_index/all"
    dirs = ["train", "validation", "test"]
    years_range = range(2013, 2023)

    processor = FileProcessor(input_dir, output_dir, dirs, years_range)
    processor.process_files()
