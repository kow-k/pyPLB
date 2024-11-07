### Functions

##
def as_tuple (L: list) -> tuple:
    "convert a list into a tuple"
    #return (*L,)
    return tuple(L)

##
def as_label (T: (list, tuple), sep: str = "", add_sep_at_end: bool = False) -> str:
    "convert a given tuple to a string by concatenating its elements"
    result = ""
    result = sep.join(T)
    if add_sep_at_end:
        result = result + sep
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

##
def count_items (L: list, item: str, check: bool = False) -> int:
    "returns the number of items in the given list"
    return len([ x for x in L if x == item ])

## parallel filter, or pfilter
def mp_filter (boolean_func, L: list):
    #from multiprocessing import Pool
    import os
    from multiprocess import Pool
    cores = max(os.cpu_count(), 1)
    with Pool (cores) as pool:
        boolean_res = pool.map (boolean_func, L)
        return [ x for x, b in zip (L, boolean_res) if b ]

##
def mp_test_for_membership (item, L: (list, tuple))-> bool:
    "multiprocess-version of membership test: effective only with a large list"
    import os
    import multiprocess as mp
    cores = max(os.cpu_count(), 1)
    with mp.Pool(cores) as pool:
        result = pool.map(lambda x: x == item, L)
    if sum(filter(lambda x: x == True, result)) > 0:
        return False
    else:
        return True


### end of file
