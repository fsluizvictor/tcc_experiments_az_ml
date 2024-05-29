import csv
import os
from typing import List

from utils import RANGE_YEARS_TEST, RANGE_YEARS_TRAIN, VREX_ENCODER_TFIDF, NEW_FILE_BASE_PATH

def main():
    years_to_train = [year for year in range(RANGE_YEARS_TRAIN[0],RANGE_YEARS_TRAIN[1] + 1)]
    years_to_test = [year for year in range(RANGE_YEARS_TEST[0],RANGE_YEARS_TEST[1] + 1)]

    _batch_csv(csv_file_path=VREX_ENCODER_TFIDF, years_range=years_to_train)
    _batch_csv(csv_file_path=VREX_ENCODER_TFIDF, years_range=years_to_test)

def _batch_csv(csv_file_path : str, years_range : List[int]):
    if not os.path.isfile(csv_file_path):
        return
    with open(csv_file_path, 'r', newline='') as arquivo_csv:
        csv_read = csv.reader(arquivo_csv)
        
        header = next(csv_read)

        csv_name = ""
        for year in years_range:
            csv_name += str(year) + "_"
        
        new_file = f"{NEW_FILE_BASE_PATH}{csv_name}.csv"
        with open(new_file, 'w', newline='') as new_csv:
            writer_csv = csv.writer(new_csv)
            writer_csv.writerow(header)

            # Loop através das linhas do arquivo de entrada
            for row in csv_read:
                cve = row[0]
                cve_year = cve.split("-")[1]
                if _contains(years_range,int(cve_year)):
                    writer_csv.writerow(row)
                else:
                    exit

def _contains(years_range: List[int], cve_year: int) -> bool:
    for year in years_range:
        if year == cve_year:
            return True

if __name__ == "__main__":
    main()
