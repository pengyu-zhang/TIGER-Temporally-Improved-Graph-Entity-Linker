import json
from collections import OrderedDict
import time
from typing import Dict


class DuplicateRemover:
    def __init__(self, input_base_dir: str, output_base_dir: str, directories: list, years: range):
        self.input_base_dir = input_base_dir
        self.output_base_dir = output_base_dir
        self.directories = directories
        self.years = years

    @staticmethod
    def remove_duplicates(input_file: str, output_file: str) -> None:
        unique_lines = OrderedDict()

        with open(input_file, 'r', encoding='utf-8') as input_f:
            for line in input_f:
                json_line = json.loads(line, object_pairs_hook=OrderedDict)
                json_str = json.dumps(json_line, ensure_ascii=False)
                if json_str not in unique_lines:
                    unique_lines[json_str] = None

        with open(output_file, 'w', encoding='utf-8') as output_f:
            for unique_line in unique_lines.keys():
                output_f.write(unique_line + '\n')

    def process_files(self):
        start_time = time.time()
        print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

        for directory in self.directories:
            for year in self.years:
                input_file = f"{self.input_base_dir}/{year}_{directory}.json"
                output_file = f"{self.output_base_dir}/{year}_{directory}.json"
                self.remove_duplicates(input_file, output_file)

        end_time = time.time()
        print(f"Completed at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
        print(f"Time cost: {(end_time - start_time) / 60:.2f} minutes\n")


if __name__ == "__main__":
    input_dir = "D:/datasets/06_dataset_7_4/01_documents"
    output_dir = "D:/datasets/06_dataset_7_4/02_documents_no_duplicates"
    dirs = ["train", "validation", "test"]
    years_range = range(2013, 2023)

    remover = DuplicateRemover(input_dir, output_dir, dirs, years_range)
    remover.process_files()
