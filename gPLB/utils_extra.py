
##
def drop_edge (self, side: str, check: bool = False):
    if side in [ "left", "Left", "L" ]:
        paired = self.paired[1:]
    elif side in  [ "right", "Right", "R" ]:
        paired = self.paired[:-1]
    else:
        raise "unrecognized side"
    ##
    p = Pattern([])
    p.paired = paired
    p.update_form()
    p.update_content()
    return p


##
def count_content(self):
    #return len([x for x in self.paired if x[0] != self.gap_mark and not self.boundary_mark in x[1] ])
    return len([ x for x in self.content if not self.boundary_mark in x ])

##
def merge_patterns (self, other, track_content: bool = False, reduction: bool = True, check: bool = False):
    "take a pair of Patterns, merges one Pattern with another"
    ## prevents void operation
    #if self.form == other.form:
    if self == other:
        return self
    if check:
        print(f"#=====================")
        print(f"#self: {self}")
        print(f"#other: {other}")
    ## main
    gap_mark      = self.gap_mark
    boundary_mark = self.boundary_mark
    ## The following two lines fail due to "TypeError: 'zip' object is not subscriptable"
    form_pairs    = list(zip (self.form, other.form))
    content_pairs = list(zip (self.content, other.content))
    if check:
        print(f"#form_pairs :{form_pairs}")
        print(f"#content_pairs: {content_pairs}")#
    ##
    new_paired = merge_patterns_main (form_pairs, content_pairs, gap_mark, boundary_mark, track_content = track_content, check = check)
    if check:
        print(f"#new_paired: {new_paired}")
    ##
    new = Pattern([])
    new.paired = new_paired
    new.update_form()
    new.update_content()
    return new


##
#@jit(nopython = True)
def merge_lattice_main (nodes, check: bool = False) -> list:
    "takes a pair of pattern lattices and returns their merger"
    #import itertools # This code needs to be externalized
    #
    merged_nodes = [ ]
    for A, B in itertools.combinations (nodes, 2):
        C = A.merge_patterns (B, check = check)
        ## The following fails to work if Pattern.__eq__ is not redefined
        #if not C in merged_nodes: # This fails.
        if is_None_free (C) and not C in merged_nodes:
            merged_nodes.append(C)
    return merged_nodes
