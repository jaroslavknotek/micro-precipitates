def param_dict_from_param_string(param_string):
    key_vals = param_string.split('-')
    pairs = [kv.split('=') for kv in key_vals]
    return dict(pairs)
    
