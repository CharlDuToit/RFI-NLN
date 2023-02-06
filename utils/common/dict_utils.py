

def to_dict(obj):
    result = {}
    for member in dir(obj):
        if member[0] != '_':
            result[member] = getattr(obj, member)
    return result

def dict_to_str(dic, seperator, *keys):
    result = ''
    for k in keys:
        result += f'{k}: {dic[k]}' + seperator
    return result



