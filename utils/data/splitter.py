
def split(splt, *args):

    if splt == 0.0 or splt == 1.0:
        return tuple( [a for a in args] + [None] * len(args))
        # raise ValueError('Split must be less than 1 and more than 0')

    lens = [len(a) for a in args if a is not None]
    for le in lens:
        if le != lens[0]:
            raise ValueError('Passed args do not have the same lenghts')
    if len(lens) == 0:
        return tuple([None] * len(args) * 2)

    le = lens[0]

    n_val = int(le * splt)
    train_set = []
    for a in args:
        if a is None:
            train_set.append(None)
        else:
            train_set.append(a[n_val:])

    val_set = []
    for a in args:
        if a is None:
            val_set.append(None)
        else:
            val_set.append(a[:n_val])

    return tuple(train_set + val_set)


if __name__ == '__main__':
    a = [0,1,2,3,4,5,6,7,8,9]
    b = [9,8,7,6,5,4,3,2,1,0]
    c = [0,1,2,3]
    a_t, b_t, a_v, b_v = split(0.2, a, b)
    a_t, b_t, a_v, b_v = split(0.2, a, None)
    a_t, b_t, a_v, b_v = split(0.2, None, None)
    a_t, b_t, a_v, b_v = split(0.2, a, b, c)





