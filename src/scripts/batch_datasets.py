import csv
import os
from typing import List

def batch_csv(csv_file_path : str, years_range : List[int]):
    # Verifica se o arquivo de entrada existe
    if not os.path.isfile(csv_file_path):
        print(f"O arquivo {csv_file_path} não existe.")
        return

    # Abre o arquivo de entrada para leitura
    with open(csv_file_path, 'r', newline='') as arquivo_csv:
        csv_read = csv.reader(arquivo_csv)
        
        # Lê o cabeçalho do arquivo
        header = next(csv_read)

        # Inicializa variáveis
        numero_arquivo = 1
        count_lines = 0
        
        csv_name = ""
        for year in years_range:
            csv_name += str(year) + "_"
        
        # Cria o primeiro arquivo de saída
        new_file = f"/home/luiz/repos/tcc_experiments_az_ml/data/samples/vrex_{csv_name}.csv"
        with open(new_file, 'w', newline='') as new_csv:
            writer_csv = csv.writer(new_csv)
            writer_csv.writerow(header)

            # Loop através das linhas do arquivo de entrada
            for row in csv_read:
                cve = row[1]
                cve_year = cve.split("-")[1]
                if _contains(years_range,int(cve_year)):
                    writer_csv.writerow(row)
                else:
                    exit

    print(f"A divisão do arquivo foi concluída. Foi criado um novo arquivo CSV para o ano {year}.")

# Exemplo de uso
csv_file_path = "/home/luiz/repos/tcc_experiments_az_ml/data/vrex.csv"
years = [year for year in range(1999, 2019)]
years.sort(reverse=True)

count = 0
batch = []
batchs = []

for year in years:
    count = count + 1
    batch.append(year)
    if count >= 10 or year == years[len(years) - 1]:
        count = 0
        batchs.append(batch)
        batch = []   

def _contains(years_range: List[int], cve_year: int) -> bool:
    for year in years_range:
        if year == cve_year:
            return True

for batch in batchs:
    batch_csv(csv_file_path,batch)
