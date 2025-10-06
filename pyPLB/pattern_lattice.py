## imports libraries

## import related modules
try:
    from .utils import *
except ImportError:
    from utils import *
try:
    from .pattern import *
except ImportError:
    from pattern import *
try:
    from .pattern_link import *
except ImportError:
    from pattern_link import *

## parameters
make_links_safely = True # False previously
debugged  = True


### Data
from dataclasses import dataclass

@dataclass
class NodeAttrs:
    rank: int
    size: int
    gap_size: int
    moment: float
    zscore: float

### Functions

def add_node_with_attrs (node_name, attrs: NodeAttrs, Gx, check: bool = False):
    """
    a routine for adding a node with attributes to Gx
    """
    if check:
        print(f"#adding node: {node_name}")
    ## to use dataclass
    Gx.add_node (node_name, **vars(attrs))

##
def as_label (T: (list, tuple), sep: str = "", add_sep_at_end: bool = False) -> str:
    """
    convert a given tuple to a string by concatenating its elements
    """

    result = ""
    result = sep.join(T)
    if add_sep_at_end:
        result = result + sep
    ##
    return result

##
def classify_pairs (r: list, l: list, check: bool = False) -> list:
    '''
    tests if a given pair is in IS-A relation and returns classification result as a list of pairs
    '''
    gap_mark = r.gap_mark
    r_form, l_form = r.form, l.form
    r_size, l_size = len (r_form), len (l_form)
    r_rank = get_rank_of_list (r_form, gap_mark)
    l_rank = get_rank_of_list (l_form, gap_mark)
    ##
    sub_pairs = []
    seen      = []
    def register_pair (p, sub_pairs = sub_pairs, seen = seen):
        if len (p) > 0 and not p in sub_pairs:
            sub_pairs.append (p)
        if len (p) > 0 and not p in seen:
            seen.append (p)
    ##
    if abs (l_size - r_size) > 1:
        if check:
            print(f"#is-a:F0; {l.form} ~~ {r.form}")
        return None
    ##
    elif l_size == r_size + 1:
        if l_form[:-1] == r_form and l_form[-1] == gap_mark:
            if check:
                print(f"#is-a:T1a; {l.form} -> {r.form}")
            register_pair ((l, r))
        elif l_form[1:] == r_form and l_form[0] == gap_mark:
            if check:
                print(f"#is-a:T1b; {l.form} -> {r.form}")
            register_pair ((l, r))
        else:
            if check:
                print(f"#is-a:F1; {l.form} -> {r.form}")
            return None
    ##
    elif l_size == r_size:
        if l_form == r_form:
            if check:
                print(f"#is-a:F2; {l.form} ~~ {r.form}")
            return None
        if r.count_gaps() == 0 and l.count_gaps() == 1:
            if r.includes (l):
                if check:
                    print(f"#is-a:T0:instance' {l.form} -> {r.form}")
                register_pair ((l, r))
            else:
                if check:
                    print(f"#is-a:F3; {l.form} ~~ {r.form}")
                return None
        #elif check_for_instantiation (r, l, check = False):
        elif r.instantiates_or_not (l, check = False):
            if check:
                print(f"#is-a:T2; {l.form} -> {r.form}")
            register_pair ((l, r))
        else:
            if check:
                print(f"#is-a:F3; {l.form} ~~ {r.form}")
            return None
    ##
    else:
        if check:
            print(f"#is-a:F4; {l.form} ~~ {r.form}")
        return None
    ## return result
    return sub_pairs

##
def classify_relations (R, L, check: bool = False):
    """
    takes two Patterns, classify their relation and returns the list of is-a cases.

    in logging, "is-a:Ti" means is-a relation is True of case i; "is-a:Fi" means is-a relation is False of case i;
    """

    sub_links = [ ]
    seen      = [ ]
    def register_link (link, sub_links = sub_links, seen = seen):
        if len(link) > 0 and not link in sub_links and not link in seen:
            sub_links.append (link)
            seen.append (link)
    ##
    for r in sorted (R, key = lambda x: len(x)):
        for l in sorted (L, key = lambda x: len(x)):
            gap_mark = r.gap_mark
            r_form, l_form = r.form, l.form
            r_size, l_size = len(r_form), len(l_form)
            r_rank = get_rank_of_list (r_form, gap_mark)
            l_rank = get_rank_of_list (l_form, gap_mark)
            ##
            if abs(l_size - r_size) > 1:
                if check:
                    print(f"#is-a:F0; {l.form} ~~ {r.form}")
                continue
            ##
            elif l_size == r_size + 1:
                if l_form[0] == gap_mark and l_form[1:] == r_form:
                    if check:
                        print(f"#is-a:T1a; {l.form} -> {r.form}")
                    register_link (PatternLink ((l, r)))
                elif  l_form[-1] == gap_mark and l_form[:-1] == r_form:
                    if check:
                        print(f"#is-a:T1b; {l.form} -> {r.form}")
                    register_link (PatternLink ((l, r)))
                else:
                    if check:
                        print(f"#is-a:F1; {l.form} -> {r.form}")
                    continue
            ##
            elif l_size == r_size:
                if r.form == l.form:
                    if check:
                        print(f"#is-a:F2; {l.form} ~~ {r.form}")
                    continue
                if r.count_gaps() == 0 and l.count_gaps() == 1:
                    if r.includes(l):
                        if check:
                            print(f"#is-a:T0:instance' {l.form} -> {r.form}")
                        register_link (PatternLink ((l, r)))
                    else:
                        if check:
                            print(f"#is-a:F3; {l.form} ~~ {r.form}")
                        continue
                #elif check_for_instantiation (r, l, check = False):
                elif r.instantiates_or_not (l, check = False):
                    if check:
                        print(f"#is-a:T2; {l.form} -> {r.form}")
                    register_link (PatternLink ((l, r)))
                else:
                    if check:
                        print(f"#is-a:F3; {l.form} ~~ {r.form}")
                    continue
            ##
            else:
                if check:
                    print(f"#is-a:F4; {l.form} ~~ {r.form}")
                continue
    ##
    return sub_links


##
def classify_relations_mp1 (R, L, gap_mark, check: bool = False):
    """
    takes two Patterns, classify their relation and returns the list of is-a cases.

    in logging, "is-a:Ti" means is-a relation is True of case i; "is-a:Fi" means is-a relation is False of case i;
    """
    ## generates a list of Boolean values
    R2 = sorted (R, key = lambda x: len(x), reverse = False)
    L2 = sorted (L, key = lambda x: len(x), reverse = False)
    from itertools import product
    #pairs = product (R2, L2) # cannot be reused
    import multiprocess as mp
    import os
    with mp.Pool(max(os.cpu_count(), 1)) as pool:
        test_values = pool.starmap (test_pairs_for_ISA, product (R2, L2))
    #print (f"test_values: {test_values}")
    true_pairs = [ pair for pair, t in zip ([ (l, r) for r, l in product (R2, L2) ], test_values) if t is True ]
    #print(f"true_pairs: {true_pairs}")
    ## remove duplicates
    links = [ PatternLink (p) for p in simplify_list (true_pairs) if len(p) == 2 ]
    #print (links)
    ##
    return links

##
def classify_relations_mp2 (R, L, gap_mark: str, check: bool = False):
    ## generates a list of Boolean values
    from itertools import product
    R2 = sorted (R, key = lambda x: len(x), reverse = False)
    L2 = sorted (L, key = lambda x: len(x), reverse = False)
    #pairs = product (R2, L2) # cannot be reused
    import os
    import multiprocess as mp
    with mp.Pool(max(os.cpu_count(), 1)) as pool:
        sub_pairs = pool.starmap (classify_pairs, product (R2, L2)) # seems to work
    ##
    return [ PatternLink (*p) for p in sub_pairs if p is not None and len(p) > 0 ] # Crucially, len(p) > 0

##
def make_ranked_dict (L: list, gap_mark: str, tracer: str) -> dict:
    "takes a list of lists and returns a dict whose keys are ranks of the lists"
    ##
    ranked_dict = {}
    for rank in set([ get_rank_of_list (x, gap_mark) for x in L ]):
        ranked_dict[rank] = [ x for x in L if Pattern(x, gap_mark, tracer).get_rank() == rank ]
    return ranked_dict

## alises
#group_nodes_by_rank = make_ranked_dict # Turned out truly offensive!

## The following is needed independently of make_ranked_dict(..)
def make_links_ranked (L: list, safely: bool, check: bool = False) -> list:
    """
    takes a list of PatternLinks and returns a dictionary of {rank: [link1, link2, ...]}
    """

    ranked_links = {}
    if safely:
        for link in L:
            if check:
                print(f"type(link): {type(link)}")
            try:
                rank = link.get_link_rank (use_max = False)
                try:
                    if not link in ranked_links[rank]:
                        ranked_links[rank].append(link)
                except KeyError:
                    ranked_links[rank] = [link]
            except AttributeError:
                print (f"failed link: {link}")
    else:
        for link in L:
            rank = link.get_link_rank (use_max = False)
            try:
                if not link in ranked_links[rank]:
                    ranked_links[rank].append(link)
            except KeyError:
                ranked_links[rank] = [link]
    ##
    return ranked_links

##
def cautiously_merge (A, B, check = False):

    C = A.merges_with (B, check = check)
    #if C is not None: # fails to work
    if len(C) > 0:
        return C # yield C fails

##
def get_rank_dists (link_dict: dict, ranked_links: dict, check: bool = False) -> dict:
    "calculate essential statistics of the rank distribution given"
    ##
    if check:
        print(f"#ranked_links: {ranked_links}")
    ##
    rank_dists = {}
    for rank in ranked_links:
        stats = {}
        members = ranked_links[rank]
        if check:
            print(f"#members: {members}")
        stats['n_members'] = len(members)
        if check:
            print(f"#n_members: {n_members}")
        dist = [ link_dict[m] for m in members ]
        if check:
            print(f"dist: {dist}")
        stats['dist'] = dist
        ##
        rank_dists[rank] = stats
    ##
    return rank_dists

##
def calc_averages_by_rank (link_dict: dict, ranked_links: dict, check: bool = False) -> dict:
    "calculate averages per rank"

    if check:
        print(f"#ranked_links: {ranked_links}")
    ##
    averages_by_rank = {}
    for rank in ranked_links:
        members = ranked_links[rank]
        dist = [ link_dict[m] for m in members ]
        averages_by_rank[rank] = sum(dist)/len(dist)
    ##
    return averages_by_rank

def calc_stdevs_by_rank (link_dict: dict, ranked_links: dict, check: bool = False) -> dict:
    "calculate stdevs per rank"
    if check:
        print(f"#ranked_links: {ranked_links}")
    ##
    import numpy as np
    stdevs_by_rank = {}
    for rank in ranked_links:
        members = ranked_links[rank]
        dist = [ link_dict[m] for m in members ]
        stdevs_by_rank[rank] = np.std(dist)
    ##
    return stdevs_by_rank

##
def calc_medians_by_rank (link_dict: dict, ranked_links: dict, check: bool = False) -> dict:
    "calculate stdevs per rank"
    if check:
        print(f"#ranked_links: {ranked_links}")
    ##
    import numpy as np
    medians_by_rank = {}
    for rank in ranked_links:
        members = ranked_links[rank]
        dist = [ link_dict[m] for m in members ]
        medians_by_rank[rank] = np.median(dist)
    ##
    return medians_by_rank

##
def calc_MADs_by_rank (link_dict: dict, ranked_links: dict, check: bool = False) -> dict:
    "calculate stdevs per rank"
    if check:
        print(f"#ranked_links: {ranked_links}")
    ## JIT compiler demand function-internal imports to be externalized
    import numpy as np
    import scipy.stats as stats
    ##
    MADs_by_rank = {}
    for rank in ranked_links:
        members = ranked_links[rank]
        dist = [ link_dict[m] for m in members ]
        MADs_by_rank[rank] = np.median (stats.median_abs_deviation (dist))
    ##
    return MADs_by_rank

##
def calc_zscore (value: float, average: float, stdev: float, median: float, MAD: float, robust: bool = True) -> float:
    "returns the z-scores of a value against average, stdev, median, and MAD given"
    ##
    import numpy as np
    import scipy.stats as stats
    robust_coeff     = 0.6745
    ##
    if stdev == 0 or MAD == 0:
        return 0
    else:
        if robust:
            return (robust_coeff * (value - median)) / MAD
        else:
            return (value - average) / stdev

##
def calc_zscore_old (value: float, average_val: float, stdev_val: float) -> float:
    "returns z-score given a triple of value, average and stdev"
    if stdev_val == 0:
        return 0
    else:
        return (value - average_val) / stdev_val

##
def normalize_zscore (x: float, use_robust_zscore: bool = False, min_val: float = -2, max_val: float = 2) -> float:
    "takes a value in the range of min, max and returns its normalized value"
    ##
    import matplotlib.colors as colors
    ## re-base when robust z-score is used
    if use_robust_zscore:
        max_val = round (1.5 * max_val, 0)
    ##
    normalizer = colors.Normalize (vmin = min_val, vmax = max_val)
    return normalizer (x)

##
def gen_zscores_from_sources (M, gap_mark: str, tracer: str, use_robust_zscore: bool, check: bool = False):
    ## adding link source z-scores to M
    Link_sources     = M.link_sources
    if check:
        print(f"##Link_sources")
    ranked_links     = make_ranked_dict (Link_sources, gap_mark = gap_mark, tracer = tracer)
    averages_by_rank = calc_averages_by_rank (Link_sources, ranked_links) # returns dict
    stdevs_by_rank   = calc_stdevs_by_rank (Link_sources, ranked_links) # returns dict
    medians_by_rank  = calc_medians_by_rank (Link_sources, ranked_links) # returns dict
    MADs_by_rank     = calc_MADs_by_rank (Link_sources, ranked_links) # returns dict

    source_zscores = {}
    for i, link_source in enumerate (Link_sources):
        value  = Link_sources[link_source]
        rank   = get_rank_of_list (link_source, gap_mark = gap_mark)
        if use_robust_zscore:
            zscore = calc_zscore (value, averages_by_rank[rank], stdevs_by_rank[rank], medians_by_rank[rank], MADs_by_rank[rank], robust = True)
        else:
            zscore = calc_zscore (value, averages_by_rank[rank], stdevs_by_rank[rank], medians_by_rank[rank], MADs_by_rank[rank], robust = False)
        ##
        source_zscores[link_source] = zscore
        if check:
            print(f"#source {i:3d}: {link_source} has {value} out-going link(s) [{source_zscores[link_source]: .4f} at rank {rank}]")

    ## attach source_zscores to M
    M.source_zscores.update(source_zscores)
    if check:
        print(f"M.source_zscores: {M.source_zscores}")
    ##
    #return M

##
def gen_zscores_from_targets (M, gap_mark: str, tracer: str, use_robust_zscore: bool, check: bool = False):
    ## adding link target z-scores to M
    Link_targets     = M.link_targets
    if check:
        print(f"##Link_targets")
    ranked_links     = make_ranked_dict (Link_targets, gap_mark = gap_mark, tracer = tracer)
    averages_by_rank = calc_averages_by_rank (Link_targets, ranked_links) # returns dict
    stdevs_by_rank   = calc_stdevs_by_rank (Link_targets, ranked_links) # returns dict
    medians_by_rank  = calc_medians_by_rank (Link_targets, ranked_links) # returns dict
    MADs_by_rank     = calc_MADs_by_rank (Link_targets, ranked_links) # returns dict

    target_zscores = {}
    for i, link_target in enumerate(Link_targets):
        value  = Link_targets[link_target]
        rank   = get_rank_of_list (link_target, gap_mark = gap_mark)
        if use_robust_zscore:
            zscore = calc_zscore (value, averages_by_rank[rank], stdevs_by_rank[rank], medians_by_rank[rank], MADs_by_rank[rank], robust = True)
        else:
            zscore = calc_zscore (value, averages_by_rank[rank], stdevs_by_rank[rank], medians_by_rank[rank], MADs_by_rank[rank], robust = False)
        target_zscores[link_target] = zscore
        if check:
            print(f"#target {i:3d}: {link_target} has {value} in-coming link(s) [{target_zscores[link_target]: .4f} at rank {rank}]")

    ## attach source_zscores to M
    M.target_zscores.update(target_zscores)
    if check:
        print(f"M.target_zscores: {M.target_zscores}")
    ##
    #return M

def gen_G (N, zscores, zscore_lb, zscore_ub, use_robust_zscore: bool, use_directed_graph: bool, test: bool = True, check: bool = False):
    """
    generate a NetworkX graph G from a given nodes N
    """

    ## modules to use
    import networkx as nx
    import math

    ## define a graph object
    if use_directed_graph:
        G = nx.DiGraph()
    else:
        G = nx.Graph() # does not accept connectionstyle specification

    ## main
    instances = [ ] # register instances
    pruned_node_count = 0

    for rank, links in sorted (N, reverse = False): # be careful on list up direction
        assert rank >= 0
        for link in links:
            if check:
                print(f"#adding link at rank {rank}: {link}")

            ## get variables
            gap_mark      = link.gap_mark

            ## get patterns for node1 and node2
            node1_p = link.left
            node2_p = link.right

            ## get sizes for node1 and node2
            node1_size = node1_p.get_size()
            node2_size = node2_p.get_size()

            ## set ranks for node1 and node2
            #node1_rank = node1_p.rank # harmful
            #node2_rank = node2_p.rank # harmful
            node1_rank = node1_p.get_rank()
            node2_rank = node2_p.get_rank()

            ## set gap sizes for node1 and node2
            #node1_gap_size = node1_p.gap_size # harmful
            #node2_gap_size = node2_p.gap_size # harmful
            node1_gap_size = node1_p.get_gap_size()
            node2_gap_size = node2_p.get_gap_size()

            ## set moment
            node1_moment = math.log(node1_size + 2)/math.log(node1_rank + 2)
            node2_moment = math.log(node2_size + 2)/math.log(node2_rank + 2)

            ## node1, node2 are node names and need to be tuples
            node1, node2  = map (tuple, link.form_paired)

            ## register node for instances
            if node1_gap_size == 0 and node1 not in instances:
                instances.append (node1)
            if node2_gap_size == 0 and node2 not in instances:
                instances.append (node2)

            ## get z-scores for node1 and node2
            try:
                node1_zscore = zscores[node1]
            except KeyError:
                node1_zscore = 0
            try:
                node2_zscore = zscores[node2]
            except KeyError:
                node2_zscore = 0

            ## Create node attributes
            node1_attrs = NodeAttrs (node1_rank, node1_size, node1_gap_size, node1_moment, node1_zscore)
            node2_attrs = NodeAttrs (node2_rank, node2_size, node2_gap_size, node2_moment, node2_zscore)

            ## add nodes and edges to G
            ## case 1: either lowerbound nor upperbound is applied
            if zscore_ub is None and zscore_lb is None:
                ## node1
                if not node1 in G.nodes():
                    add_node_with_attrs(node1, node1_attrs, G)
                else:
                    print(f"#ignored existing node {node1}")
                ## node2
                if not node2 in G.nodes():
                    add_node_with_attrs(node2, node2_attrs, G)
                else:
                    print(f"#ignored existing node {node2}")

            ## when lowerbound and upperbound z-score pruning is applied
            ## case 2: both lowerbound and upperbound
            elif zscore_ub is not None and zscore_lb is not None:
                ## node1
                if node1_zscore >= zscore_lb and node1_zscore <= zscore_ub:
                    if not node1 in G.nodes():
                        add_node_with_attrs (node1, node1_attrs, G)
                    else:
                        print(f"#ignored existing node {node1}")
                else:
                    print(f"#pruned node {node1}")
                    pruned_node_count += 1
                ## node2
                if node2_zscore >= zscore_lb and node2_zscore <= zscore_ub:
                    if not node2 in G.nodes():
                        add_node_with_attrs (node2, node2_attrs, G)
                    else:
                        print(f"#ignored exisiting node {node2}")
                else:
                    print(f"#pruned node {node2}")
                    pruned_node_count += 1

            ## case 3: lowerbound only
            elif not zscore_lb is None and zscore_ub is None: # z-score pruning applied
                ## node1
                if node1_zscore >= zscore_lb and node1_rank == rank:
                    if not node1 in G.nodes():
                        add_node_with_attrs (node1, node1_attrs, G)
                    else:
                        print(f"#ignored exisiting node {node1}")
                else:
                    print(f"#pruned node {node1}")
                    pruned_node_count += 1
                ## node2
                if node2_zscore >= zscore_lb:
                    if not node2 in G.nodes():
                        add_node_with_attrs (node2, node2_attrs, G)
                    else:
                        print(f"#ignored existing node {node2}")
                else:
                    print(f"#pruned node {node2}")
                    pruned_node_count += 1

            ## case 4: upperbound only
            elif not zscore_ub is None and zscore_lb is None:
                ## node1
                if node1_zscore <= zscore_ub:
                    if not node1 in G.nodes():
                        add_node_with_attrs (node1, node1_attrs, G)
                    else:
                        print(f"ignored existing node {node1}")
                else:
                    print(f"pruned node {node1}")
                    pruned_node_count += 1
                ## node2
                if node2_zscore <= zscore_ub:
                    if not node2 in G.nodes():
                        add_node_with_attrs (node2, node2_attrs, G)
                    else:
                        print(f"ignored existing node {node2}")
                else:
                    print(f"pruned node {node2}")
                    pruned_node_count += 1

            ## non-existing case
            else:
                raise ValueError("An undefined situation occurred")

            ## add edge
            if node1 and node2:
                G.add_edge (node1, node2)

    ## post-process for z-score pruning
    print(f"#pruned/ignored {pruned_node_count} nodes")

    ##
    return G, instances, pruned_node_count

##
def get_node_color (node, zscore_dict, padding_val: float, use_robust_zscore: bool, check: bool = False):
    """
    generate a value for node color
    """
    try:
        zscore = zscore_dict[node]
        if check:
            print(f"#z_value: {z_value: 0.4f}")
        zscore_normalized = normalize_zscore (zscore, use_robust_zscore = use_robust_zscore)
        if check:
            print(f"#zscore_normalized: {zscore_normalized: 0.4f}")
        node_color = zscore_normalized + padding_val
    except KeyError:
        node_color = padding_val
    ##
    return node_color

##
def get_node_colors_at_once (G, zscores, instances, use_robust_zscore: bool, mark_instances: bool = False, check: bool = False):

    ## node color setting
    padding_val = 0
    if mark_instances:
        padding_val = 0.05
    ##
    node_colors = []
    for node in (G):
        ## process for mark_instances
        if node in instances:
            #node_colors.append (0)
            node_color = 0
        else:
            node_color = get_node_color(node, zscores, padding_val = padding_val, use_robust_zscore = use_robust_zscore)
        node_colors.append(node_color)
    ##
    return node_colors

##
def set_node_positions (G, layout: str, scale_factor: float, key_for_MPG: str = 'gap_size'):

    """
    set node positions for drawing
    """

    import networkx as nx
    if layout in [ 'Multipartite', 'Multi_partite', 'multi_partite', 'M', 'MP', 'mp' ]:
        layout_name = "Multi-partite"
        ## scale parameter suddenly gets crucial on 2024/10/30
        positions   = nx.multipartite_layout (G, subset_key = key_for_MPG, scale = -1)
        ## flip x-coordinates
        if key_for_MPG in ['rank']:
            positions = { node: (-x, y) for node, (x, y) in positions.items() }
    ##
    elif layout in [ 'Graphviz', 'graphviz', 'G' ] :
        layout_name = "Graphviz"
        positions   = nx.nx_pydot.graphviz_layout(G, prog = 'fdp')
    ##
    elif layout in ['arf', 'ARF' ] :
        layout_name = "ARF"
        positions   = nx.arf_layout(G, scaling = scale_factor)
    ##
    elif layout in [ 'Fruchterman-Reingold', 'Fruchterman_Reingold', 'fruchterman_reingold', 'FR']:
        layout_name = "Fruchterman-Reingold"
        positions = nx.fruchterman_reingold_layout (G, scale = scale_factor, dim = 2)
    ##
    elif layout in [ 'Kamada-Kawai', 'Kamada_Kawai', 'kamda_kawai', 'KK' ]:
        layout_name = "Kamada-Kawai"
        positions   = nx.kamada_kawai_layout (G, scale = scale_factor, dim = 2)
    ##
    elif layout in [ 'Spring', 'spring', 'Sp' ]:
        layout_name = "Spring"
        positions   = nx.spring_layout (G, k = 1.4, dim = 2)
    ##
    elif layout in [ 'Shell', 'shell' , 'Sh' ]:
        layout_name = "Shell"
        positions   = nx.shell_layout (G, scale = scale_factor, dim = 2)
    ##
    elif layout in [ 'Spiral', 'spiral', 'Spr' ]:
        layout_name = "Spiral"
        positions   = nx.spiral_layout (G, scale = scale_factor, dim = 2)
    ##
    elif layout in [ 'Spectral', 'spectral', 'Spc' ]:
        layout_name = "Spectral"
        positions   = nx.spectral_layout (G, scale = scale_factor, dim = 2)
    ##
    elif layout in [ 'Circular', 'circular', 'C' ]:
        layout_name = "Circular"
        positions   = nx.circular_layout (G, scale = scale_factor, dim = 2)
    ##
    elif layout in ['Planar', 'planar', 'P'] :
        layout_name = "Planar"
        positions   = nx.planar_layout(G, scale = scale_factor, dim = 2)
    ##
    else:
        print(f"Layout is unknown: Multi-partite (default) is used")
        layout_name = "Multi-partite"
        positions   = nx.multipartite_layout (G, subset_key = key_for_MPG, scale = -1)
    ##
    return layout_name, positions

##
def draw_graph (N: dict, layout: str, key_for_MPG: str = None, fig_size: tuple = None, node_size: int = None, label_size: int = None, label_sample_n: int = None, zscores: dict = None, use_robust_zscore: bool = False, zscore_lb = None, zscore_ub = None, scale_factor: float = 3, font_name: str = None, generalized: bool = False, more_generalized: bool = False, use_directed_graph: bool = True, reverse_direction: bool = False, mark_instances: bool = True, auto_figsizing: bool = False, test: bool = False, check: bool = False) -> None:
    """
    draw a graph from a given network data
    """

    if check:
        print(f"##N with {len(N)} keys")
        for rank, links in N:
            print(f"#rank {rank}:\n{links}")

    ##
    import networkx as nx
    import math
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    #import seaborn as sns # dependency is removed on 2025/01/07

    ## generate G
    G, instances, pruned_node_count = gen_G (N, zscores = zscores, zscore_lb = zscore_lb, zscore_ub = zscore_ub, use_robust_zscore = use_robust_zscore, use_directed_graph = use_directed_graph, check = check)

    ## color values
    node_colors = get_node_colors_at_once (G, zscores, instances, use_robust_zscore = use_robust_zscore, mark_instances = mark_instances)

    ## relabeling nodes
    ## this needs to come after color setting and before layout setting
    new_labels = { x: as_label(x, sep = " ", add_sep_at_end = True) for x in G }
    G = nx.relabel_nodes (G, new_labels, copy = False)

    ## set layout and node positions
    layout_name, positions = set_node_positions (G, layout, scale_factor = scale_factor, key_for_MPG = key_for_MPG)

    ### draw
    ## set connection
    if layout_name == "Multi-partite":
        connectionstyle = "arc, angleA=0, angleB=180, armA=50, armB=50, rad=15"
    else:
        connectionstyle = "arc"

    ## set figure size
    n_items = len(instances)
    print(f"#n_items: {n_items}")
    max_item_size = max([ len(list(x)) for x in instances ])
    print(f"#max_item_size: {max_item_size}")
    max_n_segs = max([ len(x) for x in instances ])
    print(f"#max_n_segs: {max_n_segs}")

    if fig_size is None and auto_figsizing:
        #graph_width   = max_item_size + 4 + round (9 * math.log(1 + n_items))
        #graph_height  = 3 + round (9 * max_n_segs)
        if generalized:
            width_step = 2
            height_step = 2
        else:
            width_step = 2
            height_step = 2
        graph_width   = max_item_size + 3 + round (width_step * math.log(1 + n_items))
        graph_height  = 3 + round (height_step * max_n_segs)
        fig_size = (graph_width, graph_height)
    else:
        fig_size = (10, 9) # default value
    print(f"#fig_size: {fig_size}")
    plt.figure(figsize = fig_size)

    ## set label size
    if label_size is None:
        if auto_figsizing:
            label_size = 5 + round(3 * math.log(1 + n_items))
        else:
            label_size = 7 # default value
    print(f"#label_size: {label_size}")

    ## set node size
    if node_size is None:
        if auto_figsizing:
            node_size = 6 + round (3 * math.log (1 + n_items))
        else:
            node_size = 8 # default value
    print(f"#node_size: {node_size}")

    ## set font name
    if font_name is None:
        font_family = "Sans-serif" # default value
    else:
        font_family = font_name

    ## revserse the arrows
    if use_directed_graph and reverse_direction:
        G = G.reverse(copy = False) # offensive?

    ## set colormap
    my_cmap = colormaps['coolwarm']
    ## The following requires Seaborn and made obsolete
    #my_cmap = sns.color_palette("coolwarm", 24, as_cmap = True) # Crucially, as_cmap

    ## finally draw
    use_old = True
    if use_old:
        nx.draw_networkx (G, positions,
            font_family = font_family,
            font_color = 'darkblue', # label font color
            verticalalignment = "top",
            #verticalalignment = "bottom",
            horizontalalignment = "left",
            #horizontalalignment = "right",
            min_source_margin = 13, min_target_margin = 13,
            font_size = label_size, node_size = node_size,
            node_color = node_colors, cmap = my_cmap,
            edge_color = 'gray', width = 0.1, arrowsize = 6,
            arrows = True, connectionstyle = connectionstyle,
        )
    else:
        ## Draw nodes
        nx.draw_networkx_nodes (G, positions,
        node_size = node_size,
        node_color = node_colors,
        cmap = my_cmap
        )

        ## Draw edges with your existing arrow settings
        nx.draw_networkx_edges(G, positions,
            edge_color = 'gray',
            width = 0.1,
            arrowsize = 6,
            arrows = True,
            connectionstyle = connectionstyle,
            min_source_margin = 12,  # These work here
            min_target_margin = 12
        )

        ## Create custom label positions with offset
        label_offset_x = 0.005  # Adjust these values as needed
        label_offset_y = 0.01
        label_positions = {
            node: (x + label_offset_x, y + label_offset_y)
            for node, (x, y) in positions.items()
        }

        ## Draw labels separately with full control
        nx.draw_networkx_labels(G, label_positions,
            font_family = font_family,
            font_color = 'darkblue',
            font_size = label_size,
            verticalalignment = "bottom",  # or "top", "center"
            horizontalalignment = "left"   # or "right", "center"
        )

    ## set labels used in title
    instance_labels = [ as_label (x, sep = ",") for x in instances ]
    label_count = len (instance_labels)
    if label_sample_n is not None and label_count > label_sample_n:
        new_instance_labels = instance_labels[:label_sample_n - 1]
        new_instance_labels.append("…")
        new_instance_labels.append(instance_labels[-1])
        instance_labels = new_instance_labels
    print(f"#instance_labels {label_count}: {instance_labels}")

    ##
    print(f"#key_for_MPG: {key_for_MPG}")

    ### set title
    if generalized and more_generalized:
        pl_type = "G3PL"
    elif not generalized and more_generalized:
        pl_type = "G2PL"
    elif generalized and not more_generalized:
        pl_type = "G1PL"
    elif not generalized and not more_generalized:
        pl_type = "G0PL"
    ##
    if layout_name in ['Multi-partite']:
        layout_name = f"{layout_name} [key: {key_for_MPG}]"
    if use_robust_zscore:
        title_val = f"{pl_type} (layout: {layout_name}; robust z-scores: {zscore_lb} – {zscore_ub}) built from\n{instance_labels} ({label_count} in all)"
    else:
        title_val = f"{pl_type} (layout: {layout_name}; normal z-scores: {zscore_lb} – {zscore_ub}) built from\n{instance_labels} ({label_count} in all)"
    plt.title(title_val)
    ##
    plt.show()


## Classes
class PatternLattice():
    "definition of PatternLattice class"

    ##
    def __init__ (self, pattern, generalized: bool, more_generalized: bool, reflexive: bool = True, reductive: bool = True, check: bool = False):
        "initialization of a PatternLattice"
        if check:
            print(f"pattern.paired: {pattern.paired}")
        ##
        self.origin       = pattern
        self.generalized  = generalized
        self.more_generalized = more_generalized
        self.nodes        = pattern.build_lattice_nodes (generalized = generalized, more_generalized = more_generalized, check = check)
        self.gap_mark     = self.nodes[0].gap_mark
        self.ranked_nodes = self.group_nodes_by_rank (check = check)
        ## old code
        #self.links, self.link_sources, self.link_targets = self.gen_links (reflexive = reflexive, check = check)
        self.links          = self.gen_links (reflexive = reflexive, check = check)
        self.ranked_links   = make_links_ranked (self.links, safely = make_links_safely, check = check)
        self.link_sources, self.link_targets = self.get_link_stats (check = check)
        self.source_zscores = {}
        self.target_zscores = {}

    ##
    def __len__(self):
        return (len(self.nodes), len(self.links))

    ##
    def __repr__(self):
        return f"{type(self).__name__} ({self.nodes!r})"

    ##
    def __iter__(self):
        return iter (self)
        #for x in self.nodes: yield x

    ##
    def print (self):
        out = f"{type(self).__name__} ({self.nodes!r})\n"
        out += f"{type(self).__name__} ({self.source_zscores!r})\n"
        return out

    ##
    def group_nodes_by_rank (self, check: bool = False) -> dict:
        """
        takes a list of patterns, P, and generates a dictionary of patterns grouped by their ranks
        """

        from collections import defaultdict
        gap_mark  = self.gap_mark
        nodes     = self.nodes
        size      = len(nodes)

        ## implementation using itertooks.groupby() failed
        rank_finder = lambda p: len([ x for x in p.form if len(x) > 0 and x != gap_mark ])

        ## main
        ranked_nodes = defaultdict(list) # dictionary
        for pattern in sorted (nodes, key = rank_finder):
            pattern_rank = pattern.get_rank ()
            if check:
                print(f"#rank: {pattern_rank}")
                print(f"#ranked pattern: {pattern}")
            if pattern_rank <= size:
                ranked_nodes[pattern_rank].append(pattern)
        ##
        if check:
            print(f"#ranked_nodes: {ranked_nodes}")
        return ranked_nodes

    ## generate links
    def gen_links (self: object, reflexive: bool = True, use_mp: bool = True, use_mp2 = True, check: bool = False) -> list:
        """
        takes a PatternLattice P, and generates data for for P.links
        """
        ##
        gap_mark   = self.gap_mark

        ##
        ranked_nodes = self.ranked_nodes
        if len (ranked_nodes) == 0:
            ranked_nodes = make_ranked_dict (self.nodes, tracer = tracer)
        ##
        links =  [ ]
        ranks = ranked_nodes.keys()
        for rank in sorted (ranks, reverse = False):
            selected_links = [ ]
            try:
                L = simplify_list (ranked_nodes[rank])
                if check:
                    print(f"#L rank {rank} nodes: {L}")
                R = simplify_list (ranked_nodes[rank + 1])
                if check:
                    print(f"#R rank {rank + 1} nodes: {R}")
                ## make R reflexive
                if reflexive:
                    supplement = [ ]
                    for node in L:
                        if len(node) > 0 and node not in R:
                            supplement.append (node)
                    R.extend (supplement)
                ## main
                if use_mp:
                    if use_mp2:
                        classify_relations_mp = classify_relations_mp2
                    else:
                        classify_relations_mp = classify_relations_mp1
                    selected_links = classify_relations_mp (R, L, gap_mark, check = check)
                else:
                    selected_links = classify_relations (R, L, check = check)
                #print (f"selected_links: {selected_links}")
                links.extend (selected_links)
            except KeyError:
               pass
        ##
        return links

    ##
    def update_links (self, reflexive: bool, use_mp: bool = False, check: bool = False):
        """
        takes a PatternLattice P, and updates P.links, P.link_sources and P.link_targets.
        """
        ## update links
        self.links  = self.gen_links (reflexive = reflexive, use_mp = use_mp, check = check)
        ## update ranked_links
        self.ranked_links  = make_links_ranked (self.links, safely = make_links_safely, check = check)
        ## update link_sources, link_targets
        self.link_sources, self.link_targets = self.get_link_stats (check = check)
        ## return result
        return self

    ##
    def get_link_stats (self, check: bool = False):
        """
        takes a PatternLattice P, and generate data for P.link_sources and P.link_targets
        """

        from collections import defaultdict
        link_sources, link_targets = defaultdict(int), defaultdict(int)
        seen = [ ]
        for link in sorted (self.links, key = lambda x: len(x), reverse = False):
            if not link in seen:
                l_form, r_form = tuple(link.left.form), tuple(link.right.form)
                link_sources[l_form] += 1
                #if link.rsight.count_gaps() > 0:
                #    link_targets[r_form] += 1
                link_targets[r_form] += 1
                seen.append(link)
        ## return result
        return link_sources, link_targets

    ##
    def merge_lattices (self, other, **params):
        """
        take two PatternLattices and merge them into one.
        """
        gen_links_internally = params['gen_links_internally']
        generalized          = params['generalized']
        more_generalized    = params['more_generalized']
        reductive            = params['reductive']
        reflexive            = params['reflexive']
        use_mp               = params['use_mp']
        check                = params['check']

        ## merger nodes of two pattern lattices given
        main_nodes   = [ p for p in self.nodes if len(p) > 0 ]
        nodes_to_add = [ p for p in other.nodes if len(p) > 0 ]

        ## variables
        gap_mark = main_nodes[0].gap_mark
        tracer   = main_nodes[0].tracer

        ##
        if reductive:
            pool_nodes   = simplify_list (main_nodes)
            nodes_to_add = simplify_list (nodes_to_add)
        ##
        main_nodes = make_simplest_merger (main_nodes, nodes_to_add)
        if check:
            for i, node in enumerate(main_nodes):
                print(f"#main_node {i}: {node.separated_print()}")

        ## define a new pattern lattice and elaborates it
        dummy_pattern = Pattern([], gap_mark = gap_mark, tracer = tracer, check = check)
        merged = PatternLattice (dummy_pattern, generalized = generalized, more_generalized = more_generalized, reductive = reductive, check = check)
        ##
        merged.origin        = dummy_pattern
        merged.nodes         = main_nodes
        ## The following was a seriously elusive bug
        #merged.ranked_nodes  = group_nodes_by_rank (merged.nodes, gap_mark = gap_mark)
        merged.ranked_nodes  = merged.group_nodes_by_rank (check = check)
        merged.links         =  []
        merged.link_source   =  []
        merged.link_targets  =  []

        ## generate links
        if gen_links_internally:
            merged = merged.update_links (reflexive = reflexive, use_mp = use_mp, check = check)
        ##
        if check:
            print(f"#merged lattice: {merged}")
        ##
        return merged

    ##
    def draw_network (self, generalized: bool, more_generalized: bool, zscores_from_targets: bool, layout: str = None, key_for_MPG: str = None, zscore_lb: float = None, zscore_ub: float = None, use_robust_zscore: bool = False, auto_figsizing: bool = True, fig_size: tuple = None, node_size: int = None, label_size: int = None, label_sample_n: int = None, scale_factor: float = 3, font_name: str = None, test: bool = False, check: bool = False) -> None:
        """
        draw a lattice digrams from a given PatternLattice L by extracting L.links
        """
        ##
        links  = self.links
        if check:
            print(f"#links: {links}")
        ##
        sample_pattern = self.nodes[0]
        gap_mark = sample_pattern.gap_mark
        ranked_links = make_links_ranked (links, safely = make_links_safely, check = check)
        if check:
            for rank, links in ranked_links.items():
                print(f"#links at rank {rank}:\n{links}")

        ## handle z-scores
        if zscores_from_targets:
            zscores = self.target_zscores
        else:
            zscores = self.source_zscores

        ##
        if check:
            for i, item in enumerate(zscores.items()):
                node, v = item
                print(f"node {i:4d} {node} has z-score {v:.4f}")

        ## draw PatternLattice
        draw_graph (ranked_links.items(), generalized = generalized, more_generalized = more_generalized, layout = layout, key_for_MPG = key_for_MPG, auto_figsizing = auto_figsizing, fig_size = fig_size, scale_factor = scale_factor, label_sample_n = label_sample_n, font_name = font_name, zscores = zscores, use_robust_zscore = use_robust_zscore, zscore_lb = zscore_lb, zscore_ub = zscore_ub, check = check)


### end of file
