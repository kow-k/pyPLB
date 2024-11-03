### Functions

##
def count_items (L: list, item: str, check: bool = False) -> int:
    "returns the number of items in the given list"
    return len([ x for x in L if x == item ])

##
def as_tuple (L: list) -> tuple:
    "convert a list into a tuple"
    #return (*L,)
    return tuple(L)

##
def as_label (T: tuple, sep: str = "", add_sep_at_end: bool = False) -> str:
    "convert a given tuple to a string by concatenating its elements"
    result = ""
    if add_sep_at_end:
        #for x in T:
        #    result += f"{x}{sep}"
        result = sep.join(T) + sep
    else:
        #for i, x in enumerate(T):
        #    if i < len(T) - 1:
        #        result += f"{x}{sep}"
        #    else:
        #        result += f"{x}"
        result = sep.join(T)
    #
    return result

##
def simplify_list (A: list) -> list:
    C = []
    return [ x for x in A if x is not None and x not in C ]

##
def make_simplest_list (A: list, B: list) -> list:
    "takes a list or a pair of lists and returns a unification of them without reduplication"
    C = [ ]
    for a in A:
        try:
            if len(a) > 0 and a not in C:
                C.append(a)
        except TypeError:
            pass
    for b in B:
        try:
            if len(b) > 0 and not b in C:
                C.append (b)
        except TypeError:
            pass
    ##
    return C

## alias
make_list_simplest = make_simplest_list

##
def wrapped_make_simplest_list (*args):
    import functools
    return functools.reduce (make_simplest_list, args)

## parallel filter, or pfilter
def pfilter (func, X, cores):
    #from multiprocessing import Pool
    from multiprocess import Pool
    import os
    cores = max(os.cpu_count(), 1)
    with Pool (cores) as p:
        booleans = p.map (func, X)
        return [ x for x, b in zip (X, booleans) if b ]

### end of file
