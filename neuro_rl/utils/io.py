import pickle as pk

def export_pk(object, path: str):
    with open(path, "wb") as f:
        pk.dump(object, f)

def import_pk(path: str) -> object:
    with open(path, "rb") as f:
        return pk.load(f)