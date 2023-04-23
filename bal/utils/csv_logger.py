import csv
from typing import List


class CSVLogger(object):
    def __init__(self, csv_path: str, header: List[str]) -> None:
        self.csv_path = csv_path
        self.write_csv(header)

    def write_csv(self, vals: list) -> None:
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(vals)

    def __call__(self, val: list) -> None:
        self.write_csv(val)
