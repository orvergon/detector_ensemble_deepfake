import os

def existeArquivo(caminho:str):
    if not os.path.exists(caminho):
        raise RuntimeError(f"[ERRO] Não existe arquivo no caminho {caminho}.")
