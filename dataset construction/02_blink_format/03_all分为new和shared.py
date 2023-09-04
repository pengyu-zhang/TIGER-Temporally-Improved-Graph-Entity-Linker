import jsonlines
import os
from typing import List, Dict, Any

class CategoryFileSplitter:
    def __init__(self, input_base_dir: str, output_base_dir: str, directories: List[str], years: range):
        self.input_base_dir = input_base_dir
        self.output_base_dir = output_base_dir
        self.directories = directories
        self.years = years

    def split_files_by_category(self):
        for directory in self.directories:
            for year in self.years:
                input_file = f"{self.input_base_dir}/{year}/{year}_{directory}.jsonl"
                output_file_shared = f"{self.output_base_dir}/shared/{year}/{year}_{directory}.jsonl"
                output_file_new_entities = f"{self.output_base_dir}/new_entities/{year}/{year}_{directory}.jsonl"
                self._split_single_file_by_category(input_file, output_file_shared, output_file_new_entities)

    def _split_single_file_by_category(self, input_file: str, output_file_shared: str, output_file_new_entities: str) -> None:
        with jsonlines.open(input_file) as f_in, \
            jsonlines.open(output_file_shared, mode='w') as f_out_shared, \
            jsonlines.open(output_file_new_entities, mode='w') as f_out_new_entities:

            for line in f_in:
                if line["category"] == "shared":
                    f_out_shared.write(line)
                elif line["category"] == "new_entities":
                    f_out_new_entities.write(line)

if __name__ == "__main__":
    input_dir = "D:/datasets/06_dataset_7_4/04_blink_format_with_index/all"
    output_dir = "D:/datasets/06_dataset_7_4/04_blink_format_with_index"
    dirs = ["train", "validation", "test"]
    years_range = range(2013, 2023)

    splitter = CategoryFileSplitter(input_dir, output_dir, dirs, years_range)
    splitter.split_files_by_category()
