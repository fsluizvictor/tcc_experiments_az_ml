import os

def select_first_file(path) -> str:
    """Selecione o primeiro arquivo em uma pasta, assumindo que há apenas um arquivo na pasta.
    
    Args:
        path (str): Caminho para o diretório ou arquivo a ser escolhido.
        
    Returns:
        str: Caminho completo do arquivo selecionado.
    """
    files = os.listdir(path)
    return os.path.join(path, files[0])


#ENVIROMENT VARIABLES
PEARSON_FEAT_SEL = 'pearson'
SPERMAN_FEAT_SEL = 'sperman'
INFOGAIN_FEAT_SEL = 'infogain'
GINI_FEAT_SEL = 'gini'
FEATURE_KEY = 'feature'

TARGET = 'lbl_exploits_has'