# Import necessary libraries
import os
from tqdm import tqdm
import csv
import argparse

def check_dir_has_headers(directory):
    """
    Check if all CSV files in the given directory have the required header.
    
    Args:
    directory (str): Path to the directory containing CSV files.
    
    Returns:
    bool: True if all CSV files have the header, False otherwise.
    """
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            with open(os.path.join(directory, filename), "r") as f:
                reader = csv.reader(f)
                data = list(reader)
                has_header = len(data) > 0 and data[0] == [
                    "question",
                    "choice1",
                    "choice2",
                    "choice3",
                    "choice4",
                    "answer",
                ]
                if not has_header:
                    return False
    return True

def add_header_to_csv(directory):
    """
    Add the required header to all CSV files in the given directory if it's missing.
    
    Args:
    directory (str): Path to the directory containing CSV files.
    """
    for filename in tqdm(os.listdir(directory)):
        if filename.endswith(".csv"):
            with open(os.path.join(directory, filename), "r") as f:
                reader = csv.reader(f)
                data = list(reader)
                has_header = len(data) > 0 and data[0] == [
                    "question",
                    "choice1",
                    "choice2",
                    "choice3",
                    "choice4",
                    "answer",
                ]
                if not has_header:
                    # Insert the header if it's missing
                    data.insert(
                        0,
                        ["question", "choice1", "choice2", "choice3", "choice4", "answer"],
                    )
            
            # Write the updated data back to the file
            with open(os.path.join(directory, filename), "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(data)

def main():
    """
    Main function to parse arguments and process CSV files in specified directories.
    """
    parser = argparse.ArgumentParser()
    user = os.environ.get("USER")
    parser.add_argument(
        "--directory",
        type=str,
        help="directory containing MMLU .csv files",
        default="data/",
    )
    args = parser.parse_args()
    
    # Process CSV files in the 'dev' and 'test' subdirectories
    add_header_to_csv(os.path.join(args.directory, "dev"))
    add_header_to_csv(os.path.join(args.directory, "test"))

if __name__ == "__main__":
    main()