import json

def overwrite_json(filename, json_data):
    with open(filename) as f:
        obj = json.load(f)

    obj = {**obj,**json_data}
    
    with open(filename,"w+") as of:
        json.dump(obj, of)
        
class dotdict(dict):
    """ dot.notation access to dictionary attributes """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
def dict2dot(dictionary):
    dictionary = dotdict(dictionary)
    for k,v in list(dictionary.items()):
        if isinstance(v, dict):
            dictionary[k]  = dict2dot(v)
    return dictionary