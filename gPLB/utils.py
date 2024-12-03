### Functions

## parallel filter, or pfilter
def mp_Filter (boolean_func, L: list):
    """
    multiprocess verions of Filter (..)
    """
    import os
    import multiprocess as mp
    with mp.Pool (max(os.cpu_count(), 1)) as pool:
        boolean_tests = pool.map (boolean_func, L)
    return [ r for r, t in zip (L, boolean_tests) if t ]

##
def mp_test_for_inclusion (item, L: (list, tuple))-> bool:
    "multiprocess-version of membership test: effective only with a large list"
    import os
    import multiprocess as mp
    with mp.Pool(max(os.cpu_count(), 1)) as pool:
        return any(pool.map (lambda x: x == item, L))
## alias
mp_in_test = mp_test_for_inclusion

##
def process_hyphenation (W: list):
    R = []
    for w in W:
        seg = w.split("-")
        if len (seg) > 0:
            r = []
            for i, x in enumerate (seg):
                if i == 0:
                    r.append(x)
                else:
                    r.append(f"-{x}")
            R.extend (r)
        else:
            R.append (w)
    #
    return R

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
def filter_list (F: list, A: list, check: bool = False) -> list:
    #assert len(F) <= len(A)
    R = [ ]
    for x in zip(F, A[:len(F)]):
        test, value = x[0], x[1]
        if test == 0 or test is False or test is None:
            pass
        else:
            R.append(value)
    ## return result
    if check:
        print (R)
    return R

##
def sort_remove_duplicates (L: list, initial_value: object = None) -> list:
    "takes a list and returns a list with duplicates removed"
    R    = []
    prev = initial_value
    for x in sorted (L):
        if x == prev:
            pass
        else:
            R.append(x)
        ## update prev
        prev = x
    ## return result
    return R

##
def simplify_list (A: list, use_mp: bool = False) -> list:
    C = []
    if use_mp:
        ## the following turned out to be really slow
        return [ x for x in A if x is not None and len(x) > 0 and not mp_in_test (x, C) ]
    else:
        return [ x for x in A if x is not None and len(x) > 0 and x not in C ]

#def simplify_list (A:list) -> list:
#    return remove_duplicates (A, None)

## alises
reduce_list         = simplify_list
make_list_simplest  = simplify_list

##
def make_simplest_merger (A: list, B: list) -> list:
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

## aliases
make_simplest_list  = make_simplest_merger 

##
def wrapped_make_simplest_list (*args):
    import functools
    return functools.reduce (make_simplest_list, args)

##
def count_items (L: list, item: str, check: bool = False) -> int:
    "returns the number of items in the given list"
    return len([ x for x in L if x == item ])

##
def get_rank_of_list (L: (list, tuple), gap_mark: str):
    "takes a list and returns the count of its element which are not equal to gap_mark"
    return len([ x for x in L if len(x) > 0 and x != gap_mark ])

##
def attr_is_None_free (p, attr: str) -> bool:
    "tests if pattern p has no None in attribute"
    if p is None:
        return False
    L = eval(f"p.{attr}")
    return len([ x for x in L if x is None ]) == 0

##
def form_is_None_free (p: list) -> bool:
    "tests if pattern p has no None in form"
    return attr_is_None_free (p, "form")

##
def content_is_None_free (p: list) -> bool:
    "tests if pattern p has no None in content"
    return attr_is_None_free (p, "content")

##
def pattern_is_None_free (p: list) -> bool:
    "tests if pattern p has no None in form and no None in content"
    if not form_is_None_free (p):
        return False
    if not content_is_None_free (p):
        return False
    ## other cases
    return True

#def pattern_is_None_free (p):
#    "exists for compatibility check"
#    pass


### end of file
