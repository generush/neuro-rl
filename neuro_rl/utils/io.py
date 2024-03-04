import os
import pickle as pk

def ensure_directory_exists(path: str):
    """Ensure that the directory for the given path exists."""
    directory = os.path.dirname(path)
    if not os.path.exists(directory) and directory != "":
        os.makedirs(directory, exist_ok=True)

def export_pk(object, path: str):
    ensure_directory_exists(path)
    with open(path, "wb") as f:
        pk.dump(object, f)

def import_pk(path: str) -> object:
    with open(path, "rb") as f:
        return pk.load(f)