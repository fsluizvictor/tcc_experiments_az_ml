import csv
import os
from typing import List

# Nome do arquivo de saída
NEW_FILE = "/home/luiz/repos/tcc_experiments_az_ml/data/samples/consolidated/vrex_consolidated_2018_2021.csv"

# Lista com os caminhos dos arquivos de entrada que devem ser combinados
PATHS = [
    "/home/luiz/repos/tcc_experiments_az_ml/data/samples/batch_1_years/vrex_2018.csv",
    "/home/luiz/repos/tcc_experiments_az_ml/data/samples/batch_1_years/vrex_2019.csv",
    "/home/luiz/repos/tcc_experiments_az_ml/data/samples/batch_1_years/vrex_2020.csv",
    "/home/luiz/repos/tcc_experiments_az_ml/data/samples/batch_1_years/vrex_2021.csv",
]

def main():
    _join(PATHS, NEW_FILE)
    
def _join(paths: List[str], file_path: str):
    header_written = False
    with open(file_path, 'a', newline='') as new_csv:
        writer_csv = csv.writer(new_csv)
        for path in paths:
            if not os.path.isfile(path):
                print(f"O arquivo {path} não existe.")
                continue  
            
            with open(path, 'r', newline='') as input_file:
                reader_csv = csv.reader(input_file)
                
                if not header_written:
                    header = next(reader_csv)
                    writer_csv.writerow(header)
                    header_written = True
                else:
                    next(reader_csv)
                
                for row in reader_csv:
                    writer_csv.writerow(row)

if __name__ == "__main__":
    main()
