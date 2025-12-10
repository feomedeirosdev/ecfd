# %%
from pathlib import Path

def get_project_root() -> Path:
    """
    Retorna a raiz do projeto (pasta que contém o arquivo .projroot).
    Funciona independetemente de onde o script for executado.
    """

    current = Path(__file__).resolve()

    for parent in current.parents:
        if (parent/'.projroot').exists():
            return parent
        
    raise RuntimeError("Arquivo .projroot não encontrado."
                       "Verifique se ele está na raiz do projeto.")



# %%
