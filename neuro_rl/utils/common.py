from types import SimpleNamespace

def dict_to_simplenamespace(d):
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = dict_to_simplenamespace(value)
    return SimpleNamespace(**d)