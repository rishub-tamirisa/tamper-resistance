# For every csv file in a directory, add a header to the csv file
# The format of the header should be question, choice1, choice2, choice3, choice4, answer

import os
import pandas as pd
from tqdm import tqdm
import csv
import argparse

def check_dir_has_headers(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            with open(os.path.join(directory, filename), "r") as f:
                reader = csv.reader(f)
                data = list(reader)

            has_header = len(data) > 0 and data[0] == ["question", "choice1", "choice2", "choice3", "choice4", "answer"]
            if not has_header:
                return False
    return True

def add_header_to_csv(directory):
    for filename in tqdm(os.listdir(directory)):
        if filename.endswith(".csv"):
            with open(os.path.join(directory, filename), "r") as f:
                reader = csv.reader(f)
                data = list(reader)

            has_header = len(data) > 0 and data[0] == ["question", "choice1", "choice2", "choice3", "choice4", "answer"]
            if not has_header:
                data.insert(0, ["question", "choice1", "choice2", "choice3", "choice4", "answer"])
                with open(os.path.join(directory, filename), "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(data)

def main():
    parser = argparse.ArgumentParser()
    user = os.environ.get("USER")
    parser.add_argument("--directory", type=str, help="directory containing MMLU .csv files", default="data/")
    args = parser.parse_args()
    add_header_to_csv(os.path.join(args.directory, "dev"))
    add_header_to_csv(os.path.join(args.directory, "test"))

if __name__ == "__main__":
    main()