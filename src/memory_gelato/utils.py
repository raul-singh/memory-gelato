def num_elements(*tuples):
    elements = 0
    for t in tuples:
        t_size = 1
        for dim in t:
            t_size *= dim
        elements += t_size

    return elements