##
def create_gap_inserted_versions (self, gap_content: str = "_", check: bool = False):
    "add a gap at edge of a pattern given"

    n = len(self)
    if n < 2:
        return self
    else:
        #input_p = self[:] # causes trouble
        input_p = self.copy()
    ##
    gap_mark = self.gap_mark
    gapped_seg = (gap_mark, [gap_content])
    G = [] # holder of gapped versions
    import itertools
    positions = list(range(1, n)) # all positions to insert gaps
    # generate all non-empty subsets of positions
    for r in range(1, len(positions) + 1):
        for pos_combo in itertools.combinations(positions, r):
            if check:
                print(f"pos_combo: {pos_combo}")
            #copied_p = input_p[:] # causes trouble
            copied_p = input_p.copy()
            if check:
                print(f"copied_p: {copied_p}")
            # Insert gaps from right to left to maintain indices
            for pos in reversed (pos_combo):
                if check:
                    print(f"pos: {pos}")
                gapped_p = copied_p[:pos] + [gapped_seg] + copied_p[pos:]
                ## make it Pattern, crucially
                new_paired = [ (x[0], x[1]) for x in gapped_p ]
                gapped_p = Pattern([], gap_mark = gap_mark)
                gapped_p.paired = new_paired
                gapped_p.update_form()
                gapped_p.update_content()
                if check:
                    print(f"gapped_p: {gapped_p}")
            if gapped_p not in G:
                G.append(gapped_p)
    ## return result
    return G

##
def gen_L1_generalized_nodes (L, check: bool = False):
    """
    creates nodes at level 1 generalization to a given L
    """
    G = []
    for p in L:
        for position in [ 'left', 'right', 'both' ]:
            g = p.add_gaps_around (position)
            if check:
                print(f"g: {g}")
            if g not in G:
                G.append(g)
    return G

def gen_L2_generalized_nodes (L, check: bool = False):
    """
    creates nodes at level 2 generalization to a given L
    """
    G = []
    for p in L:
        for i, g in enumerate(p.create_gap_inserted_versions ()):
            if check:
                print(f"inserting g{i}: {g}")
            if g not in G:
                G.append(g)
    return G

