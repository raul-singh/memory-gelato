from typing import Optional

LOWERCASE_MEMORY_FORMATS = [
    "kb",
    "mb",
    "gb",
    "tb"
]

def num_elements(*tuples):
    elements = 0
    for t in tuples:
        t_size = 1
        for dim in t:
            t_size *= dim
        elements += t_size

    return elements


def memory_unit(format: Optional[str] = None):

    if format == None or format == "b":
        return 1

    format = format.lower()
    return 1024**(LOWERCASE_MEMORY_FORMATS.index(format) + 1)