## Functions

def test_pairs_for_ISA (r: list, l: list, check: bool = False) -> bool:
    '''
    tests if a given pair of Patterns is in IS-A relation
    '''
    gap_mark = r.gap_mark
    r_form, l_form = r.form, l.form
    r_size, l_size = len (r_form), len (l_form)
    r_rank = get_rank_of_list (r_form, gap_mark)
    l_rank = get_rank_of_list (l_form, gap_mark)
    ##
    if abs (l_size - r_size) > 1:
        if check:
            print(f"#is-a:F0; {l.form} ~ {r.form}")
        #continue
        return False
    ##
    elif l_size == r_size + 1:
        if l_form[0] == gap_mark and l_form[1:] == r_form:
            if check:
                print(f"#is-a:T1a; {l.form} <- {r.form}")
            return True
        elif  l_form[-1] == gap_mark and l_form[:-1] == r_form:
            if check:
                print(f"#is-a:T1b; {l.form} <- {r.form}")
            return True
        else:
            if check:
                print(f"#is-a:F1; {l.form} <- {r.form}")
            return False
    ##
    elif l_size == r_size:
        if r_form == l_form:
            if check:
                print(f"#is-a:F2; {l.form} ~ {r.form}")
            return False
        if r.count_gaps() == 0 and l.count_gaps() == 1:
            if r.includes(l):
                if check:
                    print(f"#is-a:T0:instance' {l.form} <- {r.form}")
                return True
            else:
                if check:
                    print(f"#is-a:F3; {l.form} ~ {r.form}")
                return False
        elif check_for_instantiation (r, l, check = False):
            if check:
                print(f"#is-a:T2; {l.form} <- {r.form}")
            return True
        else:
            if check:
                print(f"#is-a:F3; {l.form} ~ {r.form}")
            return False
    ##
    else:
        if check:
            print(f"#is-a:F4; {l.form} ~ {r.form}")
        return False


##
def classify_relationsX (R, L, check: bool = False):

    ## preclusion
    if len(R) == 0 or len(L) == 0:
        return [ ]

    ## variables
    gap_mark = R[0].gap_mark
    sub_links = [ ]
    seen = [ ]

    ## functions
    def register_link (link, sub_links = sub_links, seen = seen):
        if len (link) > 0 and not link in sub_links and not link in seen:
            sub_links.append (link)
            seen.append (link)

    ## main outer
    for r in sorted (R, key = lambda x: len(x), reverse = False):
        for l in sorted (L, key = lambda x: len(x), reverse = False):
            l_form, r_form = l.form, r.form
            l_size, r_size = len(l.form), len(r.form)
            l_rank, r_rank = l.get_rank(), r.get_rank()
            ## main
            ## false case 0 [bulky]
            if abs(l_size - r_size) > 1:
                if check:
                    print(f"#is-a:F0; {l_form} ~~ {r_form}")
                continue
            ## cases where l_size == r_size
            elif l_size == r_size:
                ## false case 1
                if r_form == l_form:
                    if check:
                        print(f"#is-a:F1; {l_form} ~~ {r_form}")
                    continue
                ## cases where l_form != r_form
                else:
                    ## true cases where r is a one-segment elaboration
                    if l_rank == 0 and r_rank == 1:
                        print(f"#is-a:T1; {l_form} <- {r_form}")
                        link = PatternLink ((l, r))
                        register_link (link)
                    ## true cases where
                    if l.count_gaps() == 1 and r.count_gaps() == 0:
                        if r.includes(l):
                            print(f"#is-a:T2:instance; {l_form} <- {r_form}")
                            link = PatternLink ((l, r))
                            register_link (link)
                        else: # false case 2
                            if check:
                                print(f"#is-a:F2; {l_form} ~~ {r_form}")
                            continue
                    ## true cases that need further checking
                    elif test_for_is_a_relation (r, l, check = False):
                        print(f"#is-a:T3; {l_form} <- {r_form}")
                        link = PatternLink ((l, r))
                        register_link (link)
                    else: # most of the cases
                        if check:
                            print(f"#is-a:F3; {l_form} ~~ {r_form}")
                        continue
            ## cases where l is one segment longer than r
            elif l_size == r_size + 1:
                ## The following code is covered by T5
                #if l_rank == 0 and r_rank == 0:
                #    print(f"#is-a:T0; {l_form} <- {r_form}")
                #    link = PatternLink ((l, r))
                #    register_link (link)
                ##
                if (l_form[1:] == r_form and r_form[-1] != gap_mark ) or (l_form[:-1] == r_form and r_form[0] != gap_mark):
                    print(f"#is-a:T4; {l_form} <- {r_form}")
                    link = PatternLink ((l, r))
                    register_link (link)
                ##
                elif l.get_substance() == r.get_substance():
                    ## This risks overlinking but ordering is crucial
                    print(f"#is-a:T5; {l_form} <- {r_form}")
                    link = PatternLink ((l, r))
                    register_link (link)

                ## the other cases
                else:
                    if check:
                        print(f"#is-a:F6; {l_form} ~~ {r_form}")
                    continue
            else: # all other fail-safe cases
                if l.get_gap_size() == l_size and r.get_gap_size() == r_size:
                    if check:
                        print(f"#is-a:F7; {l_form} <- {r_form}")
                else:
                    if check:
                        print(f"#is-a:F8; {l_form} ~~ {r_form}")
                    continue
    ##
    return sub_links



##
def gen_links_outer (l, r, check: bool = False) -> tuple:
    "takes a link node of a pattern at left and another at right, generates links, link_sources, and link_targets"
    l_form, l_content = l.form, l.content
    r_form, r_content = r.form, r.content
    if check:
        print(f"#linking l_form: {l_form}; l_content: {l_content}")
        print(f"#linking r_form: {r_form}; r_content: {r_content}")
    ## main
    if len(l_form) == 0:
        continue
    if len(r_form) == 0:
        continue
    elif l_form == r_form:
        continue
    elif r.instantiates_or_not (l, check = check):
        print(f"#instantiate: {l.form} to {r.form}")
        link = PatternLink([l, r])

##
def mp_gen_links_main (links, link_souces, link_targets, x, check: bool = False):
    "take arguments and updates"
    #
    r, l = x[0], x[1]
    r_form, r_content = r.form, r.content
    l_form, l_content = l.form, l.content
    if check:
        print(f"#linking r_form: {r_form}; r_content: {r_content}")
    ## main
    if len(r_form) == 0 or len(l_form):
        pass
    elif l_form == r_form:
        pass
    elif r.instantiates_or_not (l, check = check):
        print(f"#instantiate {l.form} to {r.form}")
        link = PatternLink ([l, r])
        ##
        if not link in links:
            ## register for links
            links.append (link)
            ## register for link_sources, link_targets
            try:
                link_sources[l_form] += 1
                link_targets[r_form] += 1
            except KeyError:
                link_sources[l_form] = 1
                link_targets[r_form] = 1
    ## result is None

##
def X():
    ## filter insignificant nodes
    prune_count = 0
    if not zscore_lowerbound is None:
       print(f"#pruning nodes with z-score less than {zscore_lowerbound}")
       nodes_to_keep     = [ ]
       nodes_to_keep_ids = [ ]
       for i, node in enumerate(G):
           node_as_tuple = as_tuple(node)
           if check:
               print(f"#node_as_tuple: {node_as_tuple}")
           ## assign z-score
           try:
               z_value = zscores[node_as_tuple]
           except KeyError:
               z_value = 0 # default value for z_score
           if check:
               print(f"#z_value: {z_value}")
           ## filter nodes
           if zscore_lowerbound <= z_value :
               nodes_to_keep.append (node)
               nodes_to_keep_ids.append (i)
           else:
               print(f"#pruning {node} with z-score {z_value:.5f}")
               prune_count += 1
        print(f"#pruned {prune_count} nodes")

        # remove nodes
        G.remove_nodes_from ([ x for x in G if not x in nodes_to_keep ])

        # set node colors
        values_for_color_filtered = [ ]
        for i, value in enumerate(values_for_color):
           if i in nodes_to_keep_ids:
               values_for_color_filtered.append (value)
        values_for_color = values_for_color_filtered

## Method

    ##
    def merge_lattices_prev (self, other, gen_links_internally: bool, generalized: bool = True, reflexive: bool = True, reductive: bool = True, remove_None_containers: bool = False, show_steps: bool = False, use_multiprocess: bool = True, check: bool = False):
        "takes a pair of PatternLattices and returns its merger"
        ##
        print(f"#merging pattern lattices ...")
        #
        if len (self.nodes) > 0 and len (other.nodes) == 0:
            return self
        elif len (self.nodes) == 0 and len (other.nodes) > 0:
            return other
        ##
        sample_pattern = self.nodes[0]
        gap_mark       = sample_pattern.gap_mark
        import itertools # This code needs to be externalized under jit

        ## creates .nodes
        pooled_nodes  = self.nodes
        nodes_to_add  = other.nodes
        ## remove None-containers
        #if remove_None_containers:
        #    pooled_nodes = [ node for node in pooled_nodes if form_is_None_free (node) ]
        #    nodes_to_add = [ node for node in other.nodes if form_is_None_free (node) ]
        ##
        pooled_nodes = [ p for p in pooled_nodes if len(p) > 0 ]
        nodes_to_add = [ p for p in nodes_to_add if len(p) > 0 ]
        if check:
            print(f"#pooled_nodes [0]: {pooled_nodes}")
            print(f"#nodes_to_add [0]: {nodes_to_add}")

        ## reduce nodes from self
        size1 = len (pooled_nodes)
        if reductive:
            R = [ ]
            ## The following fails to work if Pattern.__eq__ is not redefined
            for node in pooled_nodes:
                if node not in R:
                    R.append (node)
            pooled_nodes = R
            ## check difference
            size2 = len(pooled_nodes)
            d = size2 - size1
            if d > 0:
                print(f"#reduced {d} nodes from pooled nodes")
        if check:
            print(f"#pooled_nodes [1]: {pooled_nodes}")

        ## adding nodes from other
        if reductive:
            for node in nodes_to_add:
                ## The following fails to work if Pattern.__eq__ is not redefined
                if not node in pooled_nodes:
                    pooled_nodes.append (node)
            if check:
                print(f"#nodes_to_add [0]: {nodes_to_add}")
        else:
            pooled_nodes += nodes_to_add

        ## merger main
        merged_nodes = [ ]
        ## multiprocess(ing) version
        if use_multiprocess:
            print(f"#running in multi-processing mode")
            import os
            import multiprocess as mp
            cores = max (os.cpu_count(), 1)
            with mp.Pool(cores) as pool:
                for size, sized_nodes in group_nodes_by_size (pooled_nodes, gap_mark = gap_mark, reverse = True).items():
                    if check:
                        print(f"#sized_nodes {size}: {sized_nodes}")
                    merged_sized_nodes = pool.starmap (merge_patterns_and_filter, itertools.combinations (sized_nodes, 2))
                    merged_sized_nodes = simplify_list (merged_sized_nodes)
                    merged_nodes.extend (merged_sized_nodes)
        ## original slower version
        else:
            for rank, sized_nodes in group_nodes_by_size (pooled_nodes, gap_mark = gap_mark, reverse = True).items():
                if check:
                    print(f"#sized_nodes {size}: {sized_nodes}")
                merged_sized_nodes = [ ]
                for A, B in itertools.combinations (sized_nodes, 2):
                    C = A.merges_with (B, check = False)
                    ## The following fails unless Pattern.__eq__ is redefined
                    if len(C) > 0 and not C in merged_sized_nodes:
                        merged_sized_nodes.append (C)
                ##
                merged_nodes.extend (merged_sized_nodes)

        ## remove None-containing nodes
        #merged_nodes = [ p for p in merged_nodes if not p is None ]
        ##
        if check:
            print(f"#merged_nodes: {merged_nodes}")

        # generate merged PatternLattice
        empty_pat            = Pattern([], gap_mark)
        merged               = PatternLattice (empty_pat, generalized = generalized, reflexive = reflexive)
        merged.nodes         = merged_nodes
        merged.ranked_nodes  = merged.group_nodes_by_rank (check = check)
        if check:
            print(f"#merged_ranked_nodes: {merged.ranked_nodes}")

        ##
        merged.links, merged.link_sources, merged.link_targets = [ ], [ ], [ ]
        ## conditionally generates links
        if gen_links_internally:
            merged.links, merged.link_sources, merged.link_targets  = \
                merged.gen_links (reflexive = reflexive, check = check)

        ## return result
        if len(merged.links) > 0:
            if show_steps:
                print(f"#merger into {len(merged_nodes)} nodes done")
        #yield merged # fails
        return merged


    ##
    def gen_ranked_links_prev (self, reflexive: bool, reductive: bool = True, check: bool = False):
        "takes a PatternLattice, extracts ranked_nodes, and generates a dictionary of links {rank: [link1, link2, ...]}"
        ##
        G = self.ranked_nodes
        ranked_links = {}
        for rank in sorted (G.keys()):
            if check:
                print(f"#rank: {rank}")
            ## define L
            L = G[rank]
            if check:
                print(f"#L: {list(L)}")
            ## define R
            try:
                R = G[rank + 1]
                ## handles reflexivity
                if reflexive:
                    R = make_simplest_list (L, R)
            except KeyError:
                pass
            ##
            for l in L:
                l_form, l_content = l.form, l.content
                for r in R:
                    r_form, r_content = r.form, r.content
                    if check:
                        print(f"#linking l_form: {l_form}; l_content: {l_content}")
                        print(f"#linking r_form: {r_form}; r_content: {r_content}")
                    ## main
                    if l_form == r_form:
                        continue
                    elif r.instantiates_or_not (l):
                        link = PatternLink([l, r])
                    else:
                        continue
                    ##
                    if link:
                        rank = link.get_rank()
                        try:
                            if link and not link in ranked_links[rank]:
                                ranked_links[rank].append(link)
                        except KeyError:
                            ranked_links[rank] = [link]
        ##
        return ranked_links

    ##
    def gen_links_X (self, reflexive: bool, reductive: bool = True, check: bool = True):
        "takes a PatternLattice, extracts ranked_nodes, and generates a list of links among them"
        ##
        if check:
            print(f"#generating links ...")
        ##
        links = [ ]
        #seen = [ ]
        #link_sources, link_targets = {}, {}
        link_sources, link_targets = defaultdict(int), defaultdict(int)
        G = self.ranked_nodes
        ## main to be multiprocessed
        for rank in sorted (G.keys()):
            if check:
                print(f"#rank: {rank}")
            ##
            L = sorted (G[rank], key = lambda x: len(x))
            if check:
                print(f"#L: {list(L)}")
            ## define R
            try:
                R = G[rank + 1]
                if check:
                    print(f"#R: {list(R)}")
            except KeyError:
                pass
            #
            if reflexive:
                R = sorted (make_simplest_list (R, L), key = lambda x: len(x))

            ## main: L and R are given
            sub_links = [ ]
            ## define L
            try:
                for l in L: # l is a Pattern
                    l_form, l_content = l.form, l.content
                    if len(l_form) == 0:
                        #print(f"#ignored l_form: {l_form}")
                        continue
                    for r in R: # r is a Pattern
                        r_form, r_content = r.form, r.content
                        assert len(l.get_substance()) <= len(r.get_substance())
                        ## main
                        if len(r_form) == 0 or l_form == r_form:
                            #print(f"#ignored pair: {l_form}; {r_form}")
                            continue
                        ## valid case
                        elif len(l_form) == len(r_form) or len(l_form) == len(r_form) + 1:
                            if r.instantiates_or_not (l, check = check):
                                print(f"#is-a: {l_form} <- {r_form}")
                                ##
                                link = PatternLink ((l, r))
                                #if mp_test_for_membership (link, links): # This slows down
                                #if not link in links:
                                if link and not link in sub_links:
                                #if not link in sub_links and not link in sub_links_pre1:
                                #if not link in seen:
                                    #links.append (link)
                                    sub_links.append (link)
                                    link_sources[l_form] += 1
                                    link_targets[r_form] += 1
                                    ## update seen
                                    #seen.append(link)
                            else:
                                #print(f"#not-is-a 2: {l_form} <-/- {r_form}")
                                continue
                        else:
                            #print(f"#not-is-a 1: {l_form} <-/- {r_form}")
                            continue
                ## update sub_links_prev
                #sub_links_pre1 = sub_links
                ##
                links.extend (sub_links)
                #links.extend ([ link for link in sub_links if not link in links ])
                if check:
                    print(f"#links.extend: {links}")
            ##
            except UnboundLocalError:
                pass

        ## return
        return links, link_sources, link_targets
        #yield links, link_sources, link_targets

    ##
    def update_links_prev (self, reflexive: bool = True, reductive: bool = True, check: bool = False):
        "update links"
        L, L_sources, L_targets = self.gen_links (reflexive = reflexive, reductive = reductive, check = check)
        if check:
            print(f"#L (in update): {L}")
        ##
        self.links = L
        self.link_sources = L_sources
        self.link_targets = L_targets
        return self


