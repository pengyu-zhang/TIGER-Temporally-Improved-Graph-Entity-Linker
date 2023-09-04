import jsonlines
import json
import time
from typing import Dict, Any
from contextlib import ExitStack


class FileProcessor:
    def __init__(self, input_base_dir: str, output_base_dir: str, directories: list, years: range):
        self.input_base_dir = input_base_dir
        self.output_base_dir = output_base_dir
        self.directories = directories
        self.years = years

    @staticmethod
    def process_line(line: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "title": line["target_title"],
            "text": line["target_bert"],
            "document_id": line["target_page_id"],
            "qid": line["target_qid"],
        }

    def process_files(self):
        start_time = time.time()
        print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

        with ExitStack() as stack:
            output_files = self._open_all_output_files(stack)
            self._process_all_input_files(output_files)

        end_time = time.time()
        print(f"Completed at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
        print(f"Time cost: {(end_time - start_time) / 60:.2f} minutes\n")

    def _open_all_output_files(self, stack):
        return {
            directory: {
                year: stack.enter_context(
                    open(
                        f"{self.output_base_dir}/{year}_{directory}.json",
                        "a",
                        encoding="utf-8",
                    )
                )
                for year in self.years
            }
            for directory in self.directories
        }

    def _process_all_input_files(self, output_files):
        for directory in self.directories:
            for year in self.years:
                with jsonlines.open(
                        f"{self.input_base_dir}/{directory}/{year}-01-01T00_00_00Z_{directory}.jsonl"
                ) as f:
                    for line in f:
                        processed_line = self.process_line(line)
                        json_str = json.dumps(processed_line, ensure_ascii=False)
                        output_files[directory][year].write(json_str + "\n")


if __name__ == "__main__":
    input_dir = "D:/datasets/06_dataset_7_4/tempel_v1.0_only_bert_tokenized"
    output_dir = "D:/datasets/06_dataset_7_4/01_documents"
    dirs = ["train", "validation", "test"]
    years_range = range(2013, 2023)

    processor = FileProcessor(input_dir, output_dir, dirs, years_range)
    processor.process_files()
