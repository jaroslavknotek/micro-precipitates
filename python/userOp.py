def userOp(text):
    parts = text.split(' ')
    if parts[1] == '+':
        return parts[0] + parts[2]
    elif parts[1]== '-':
        return parts[0] - parts[2]
    else:
        raise ValueError('Unsupported opperation.')