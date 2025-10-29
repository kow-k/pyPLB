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
debugged  = False

### Data
from dataclasses import dataclass

@dataclass
class NodeAttrs:
    literal: tuple
    size: int
    gap_size: int
    rank: int
    zscore: float
    moment: float

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
def as_label (T: (list, tuple), sep: str = " ", add_sep_at_end: bool = False) -> str:
    """
    converts a given tuple to a string by concatenating its elements
    """

    result = ""
    result = sep.join(T)
    if add_sep_at_end:
        result = result + sep
    ##
    return result

##
def register_pair (p, sub_pairs):
    "helper function for register_pairs()"
    if len (p) > 0 and not p in sub_pairs:
        sub_pairs.append (p)

##
def register_pairs (r: list, l: list, check: bool = False) -> list:
    """
    tests if a given pair is in IS-A relation and returns classification result as a list of pairs
    """
    gap_mark = r.gap_mark
    tracer   = r.tracer
    r_form, l_form = r.form, l.form
    r_size, l_size = len (r_form), len (l_form)
    r_rank = get_rank_of_list (r_form, gap_mark)
    l_rank = get_rank_of_list (l_form, gap_mark)

    ##
    size_diff = l_size - r_size
    sub_pairs = []
    if size_diff == 0:
        if isa_under_size_equality(r_form, l_form, gap_mark = gap_mark, tracer = tracer, check = check):
            register_pair ((l, r), sub_pairs)
    elif size_diff == 1:
        if isa_under_size_difference(r_form, l_form, gap_mark = gap_mark, tracer = tracer, check = check):
            register_pair ((l, r), sub_pairs)
    else:
        pass

    ## return result
    return sub_pairs

##
def classify_relations_mp2 (R, L, gap_mark: str, check: bool = False):
    """
    Alternative multiprocess version using register_pairs instead of test_pairs_for_ISA.
    """

    from itertools import product
    import os
    import multiprocess as mp

    ## sort R and L once
    R2 = sorted (R, key = lambda x: len(x), reverse = False)
    L2 = sorted (L, key = lambda x: len(x), reverse = False)

    #pairs = product (R2, L2) # cannot be reused
    pairs = list(product (R2, L2)) # can be reused; crucially, list(product(..))
    with mp.Pool(max(os.cpu_count(), 1)) as pool:
        sub_pairs = pool.starmap (register_pairs, pairs) # seems to work
    ## Flatten nested structure and create PatternLinks
    ## sub_pairs is [[pair1], [pair2], [], None, [pair3], ...]
    links = [ PatternLink(pair) for result in sub_pairs
                if result is not None and len(result) > 0
                    for pair in result if len(pair) == 2 ]
    ##
    return links

##
def test_pairs_for_ISA (r: list, l: list, check: bool = False) -> bool:
    """
    tests if a given pair of Patterns is in IS-A relation [called from classify_relations_mp1()]
    """

    gap_mark = r.gap_mark
    r_form, l_form = r.form, l.form
    r_size, l_size = len (r_form), len (l_form)
    r_rank = get_rank_of_list (r_form, gap_mark)
    l_rank = get_rank_of_list (l_form, gap_mark)
    ##
    if abs (l_size - r_size) > 1:
        if check:
            print(f"#is-a:F0; {l.form} ~ {r.form}")
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
def classify_relations_mp1 (R, L, gap_mark, check: bool = False):
    """
    takes two Patterns, classify their relation and returns the list of is-a cases.

    in logging, "is-a:Ti" means is-a relation is True of case i; "is-a:Fi" means is-a relation is False of case i;
    """
    from itertools import product
    import multiprocess as mp
    import os

    ## generates a list of Boolean values and sort once
    R2 = sorted (R, key = lambda x: len(x), reverse = False)
    L2 = sorted (L, key = lambda x: len(x), reverse = False)

    #pairs = product (R2, L2) # cannot be reused
    pairs = list(product (L2, R2)) # can be re-used; Crucially, list(product(..))
    with mp.Pool(max(os.cpu_count(), 1)) as pool:
        test_values = pool.starmap (test_pairs_for_ISA, pairs)

    ## filter in true cases
    true_pairs = [ (l, r) for (l, r), is_true in zip(pairs, test_values) if is_true ]

    ## remove duplicates
    links = [ PatternLink (p) for p in simplify_list (true_pairs) if len(p) == 2 ]
    ##
    return links


##
def register_link (link, sub_links, seen):
    "helper function for classify_relaitons_nmp"
    if len(link) > 0 and not link in sub_links and not link in seen:
        sub_links.append (link)
        seen.append (link)

##
def classify_relations_nmp (R, L, check: bool = False):
    """
    takes two Patterns, classify their relation and returns the list of is-a cases.

    in logging, "is-a:Ti" means is-a relation is True of case i; "is-a:Fi" means is-a relation is False of case i;
    """

    sub_links = [ ]
    seen      = [ ]
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
                    register_link (PatternLink ((l, r)), sub_links, seen)
                elif  l_form[-1] == gap_mark and l_form[:-1] == r_form:
                    if check:
                        print(f"#is-a:T1b; {l.form} -> {r.form}")
                    register_link (PatternLink ((l, r)), sub_links, seen)
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
                        register_link (PatternLink ((l, r)), sub_links, seen)
                    else:
                        if check:
                            print(f"#is-a:F3; {l.form} ~~ {r.form}")
                        continue
                #elif check_for_instantiation (r, l, check = False):
                #elif r.instantiates_or_not (l, check = False):
                elif l.subsumes_or_not (r, check = False):
                    if check:
                        print(f"#is-a:T2; {l.form} -> {r.form}")
                    register_link (PatternLink ((l, r)), sub_links, seen)
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
def group_links_by (metric: str, L: list, gap_mark: str, tracer: str) -> dict:
    """
    takes a list of lists and returns a dict whose keys are ranks of the lists
    """

    assert metric in ['rank', 'gap_size']
    ##
    link_dict = {}
    if metric == 'rank':
        G = [ get_rank_of_list (l, gap_mark) for l in L ]
        for metric_val in set(G):
            link_dict[metric_val] = [ l for l in L if metric_val == Pattern(l, gap_mark, tracer).get_rank() ]
    elif metric == 'gap_size':
        G = [ get_gap_size_of_list (l, gap_mark) for l in L ]
        for metric_val in set(G):
            link_dict[metric_val] = [ l for l in L if metric_val == Pattern(l, gap_mark, tracer).get_gap_size() ]
    else:
        raise ValueError("Unknown metric: should be either 'rank' or 'gap_size'")
    ##
    return link_dict

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
def make_links_ranked (L: list, safely: bool, use_max: bool = False, check: bool = False) -> list:
    """
    takes a list of PatternLinks and returns a dictionary of {rank: [link1, link2, ...]}
    """
    ranked_links = {}
    if safely:
        for link in L:
            if check:
                print(f"#type(link): {type(link)}")
            try:
                rank = link.get_link_rank (use_max = use_max)
                try:
                    if not link in ranked_links[rank]:
                        ranked_links[rank].append(link)
                except KeyError:
                    ranked_links[rank] = [link]
            except AttributeError:
                print (f"#failed to add link: {link}")
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
def make_links_grouped_by (metric: str, L: list, use_max: bool, safely: bool = False, check: bool = False) -> list:
    """
    takes a list of PatternLinks and returns a dictionary of {metric_val: [link1, link2, ...]}
    """

    if check:
        print(f"#metric: {metric}")
        print(f"#PatternLinks: {L}")
    ##
    import collections
    grouped_links = collections.defaultdict(list)
    if safely:
        for link in L:
            if check:
                print(f"#type(link): {type(link)}")
            try:
                if   metric == 'rank':
                    metric_val = link.get_link_rank (use_max = use_max)
                elif metric == 'gap_size':
                    metric_val = link.get_link_gap_size (use_max = not(use_max))
                if not link in grouped_links[metric_val]:
                    grouped_links[metric_val].append(link)
            except AttributeError:
                print (f"#failed to add link: {link}")
    else:
        for link in L:
            if check:
                    print(f"#type(link): {type(link)}")
            if   metric == 'rank':
                metric_val = link.get_link_rank (use_max = use_max)
            elif metric == 'gap_size':
                metric_val = link.get_link_gap_size (use_max = not(use_max))
            if not link in grouped_links[metric_val]:
                grouped_links[metric_val].append(link)
    ##
    return grouped_links

##
def cautiously_merge (A, B, check = False):

    C = A.merges_with (B, check = check)
    #if C is not None: # fails to work
    if len(C) > 0:
        return C # yield C fails

def merge_pattern_lattices (Ls: list, generality: int, gen_links_internally: bool = True, use_mp: bool = True, reflexive: bool = True, reductive: bool = True, check: bool = False):
    """takes a list of PatternLattices and returns its merger."""

    print(f"merging {len(Ls)} PLs...")
    import functools
    M = functools.reduce (lambda La, Lb: La.merge_with (Lb, gen_links_internally = gen_links_internally, use_mp = use_mp, generality = generality, reflexive = reflexive, reductive = reductive, check = check), Ls)

    # The following process was isolated for memory conservation
    if gen_links_internally:
        print(f"##updating links...")
        ## N.B. 1) Don't do: M = M.update(...); 2) update_links() is rank-based
        M.update_links (reflexive = reflexive, use_mp = use_mp, check = check)
    ##
    return M


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
def calc_statistics_by (metric: str, link_dict: dict, grouped_links: dict,
                       stat_func, check: bool = False) -> dict:
    """
    Calculate statistics per a given metric using stat_func.

    Args:
        metric: Either 'rank' or 'gap_size'
        link_dict: Dictionary mapping nodes to their link counts
        grouped_links: Dictionary grouping nodes by metric value
        stat_func: Function to calculate statistic or string ('mean', 'std', 'median', 'mad')
        check: If True, print debug information

    Returns:
        Dictionary mapping metric values to calculated statistics
    """
    assert metric in ['rank', 'gap_size']

    import numpy as np
    import scipy.stats as stats

    ## helper functions
    MAD_calc = lambda dist: np.median (stats.median_abs_deviation (dist))

    ## Handle string shortcuts
    if isinstance(stat_func, str):
        stat_funcs = {
            'mean'    : np.mean,
            'average' : np.mean,
            'std'     : np.std,
            'stdev'   : np.std,
            'median'  : np.median,
            'MAD'     : MAD_calc,
            'mad'     : MAD_calc,
        }
        if stat_func not in stat_funcs:
            raise ValueError(f"Unknown stat function: {stat_func}")
        stat_func = stat_funcs[stat_func]
    ##
    if check:
        print(f"#grouped_links: {grouped_links}")
    ##
    stats_by = {}
    for metric_val in grouped_links:
        members = grouped_links[metric_val]
        dist = [link_dict[m] for m in members]
        stats_by[metric_val] = stat_func(dist)
    ##
    return stats_by

# Keeping the old _by_rank versions for backward compatibility if needed
calc_averages_by_rank = lambda ld, rl, c=False: calc_statistics_by('rank', ld, rl, 'mean', c)
calc_stdevs_by_rank = lambda ld, rl, c=False: calc_statistics_by('rank', ld, rl, 'std', c)
calc_medians_by_rank = lambda ld, rl, c=False: calc_statistics_by('rank', ld, rl, 'median', c)
calc_MADs_by_rank = lambda ld, rl, c=False: calc_statistics_by('rank', ld, rl, 'mad', c)
# And for the _by versions
calc_averages_by = lambda m, ld, gl, c=False: calc_statistics_by(m, ld, gl, 'mean', c)
calc_stdevs_by = lambda m, ld, gl, c=False: calc_statistics_by(m, ld, gl, 'std', c)
calc_medians_by = lambda m, ld, gl, c=False: calc_statistics_by(m, ld, gl, 'median', c)
calc_MADs_by = lambda m, ld, gl, c=False: calc_statistics_by(m, ld, gl, 'mad', c)

##
def calc_zscore (value: float, average: float, stdev: float, median: float = None, MAD: float = None, robust: bool = True) -> float:
    """
    returns the z-scores of a value against average, stdev, median and MAD given
    """

    ##
    import numpy as np
    import scipy.stats as stats
    #robust_coeff     = 0.6745 # turned out to be wrong.
    robust_coeff     = 1.4826 # scale factor to make MAD comparable to stdev
    ##
    if stdev == 0 or MAD == 0:
        return 0
    else:
        if robust:
            return (value - median) / (MAD * robust_coeff)
        else:
            return (value - average) / stdev

##
def gen_zscores_from_targets_by (metric: str, M: object, gap_mark: str, tracer: str, use_robust_zscore: bool = True, check: bool = False) -> None:
    """
    given a PatternLattice M, creates z-scores from link targets, calculated by metric (rank or gap_size) and attach the result to M.
    """

    Link_targets = M.link_targets
    if check:
        print(f"##Link_targets")

    ## The following all return list
    grouped_links = group_links_by (metric, Link_targets, gap_mark = gap_mark, tracer = tracer)
    assert len(grouped_links)

    ##
    averages_by = calc_statistics_by (metric, Link_targets, grouped_links, 'average')
    stdevs_by   = calc_statistics_by (metric, Link_targets, grouped_links, 'stdev')
    medians_by  = calc_statistics_by (metric, Link_targets, grouped_links, 'median')
    MADs_by     = calc_statistics_by (metric, Link_targets, grouped_links, 'MAD')

    ##
    target_zscores = {}
    for i, link_target in enumerate (Link_targets):
        value  = Link_targets[link_target]
        ##
        if metric == 'rank':
            metric_val  = get_rank_of_list (link_target, gap_mark = gap_mark)
        elif metric == 'gap_size':
            metric_val  = get_gap_size_of_list (link_target, gap_mark = gap_mark)
        ##
        if use_robust_zscore:
            zscore = calc_zscore (value, averages_by[metric_val], stdevs_by[metric_val], medians_by[metric_val], MADs_by[metric_val], robust = True)
        else:
            zscore = calc_zscore (value, averages_by[metric_val], stdevs_by[metric_val], medians_by[metric_val], MADs_by[metric_val], robust = False)
        ##
        target_zscores[link_target] = zscore
        print(f"#target {i:3d}: {link_target} has {value} in-link(s) at {metric} {metric_val} [z-score: {target_zscores[link_target]:.3f}]")

    ## attach target_zscores to M
    if use_robust_zscore: # robust z-scores
        M.target_robust_zscores.update(target_zscores)
        if check:
            print(f"M.target_robust_zscores: {M.target_robust_zscores}")
    else: # normal z-scores
        M.target_zscores.update(target_zscores)
        if check:
            print(f"M.target_robust_zscores: {M.target_robust_zscores}")

##
def gen_zscores_from_sources_by (metric: str, M: object, gap_mark: str, tracer: str, use_robust_zscore: bool = True, check: bool = False) -> None:
    """
    given a PatternLattice M, creates z-scores from link sources, calculated by metric (rank or gap_size) and attach the result to M.
    """

    Link_sources = M.link_sources
    if check:
        print(f"##Link_sources")

    ## The following all return list
    grouped_links = group_links_by (metric, Link_sources, gap_mark = gap_mark, tracer = tracer)
    assert len(grouped_links)

    ##
    averages_by = calc_statistics_by (metric, Link_sources, grouped_links, 'average')
    stdevs_by   = calc_statistics_by (metric, Link_sources, grouped_links, 'stdev')
    medians_by  = calc_statistics_by (metric, Link_sources, grouped_links, 'median')
    MADs_by     = calc_statistics_by (metric, Link_sources, grouped_links, 'MAD')

    ##
    source_zscores = {}
    for i, link_source in enumerate (Link_sources):
        value  = Link_sources[link_source]
        ##
        if metric == 'rank':
            metric_val = get_rank_of_list (link_source, gap_mark = gap_mark)
        elif metric == 'gap_size':
            metric_val = get_gap_size_of_list (link_source, gap_mark = gap_mark)
        ##
        if use_robust_zscore:
            zscore = calc_zscore (value, averages_by[metric_val], stdevs_by[metric_val], medians_by[metric_val], MADs_by[metric_val], robust = True)
        else:
            zscore = calc_zscore (value, averages_by[metric_val], stdevs_by[metric_val], medians_by[metric_val], MADs_by[metric_val], robust = False)
        ##
        source_zscores[link_source] = zscore
        print(f"#source {i:3d}: {link_source} has {value} out-link(s) at {metric} {metric_val} [z-score: {source_zscores[link_source]:.3f}]")

    ## attach source_zscores to M
    if use_robust_zscore:
        M.source_robust_zscores.update(source_zscores)
        if check:
            print(f"M.source_robust_zscores: {M.source_robust_zscores}")
    else: # normal z-score
        M.source_zscores.update(source_zscores)
        if check:
            print(f"M.source_zscores: {M.source_zscores}")

##
def gen_G (N, zscores, use_directed_graph: bool, test: bool = True, check: bool = False):
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

    for key, links in sorted (N, reverse = False): # be careful on list up direction
        assert key >= 0
        for link in links:
            if check:
                print(f"#adding link at group {key}: {link}")

            ## get variables
            gap_mark      = link.gap_mark

            ## get patterns for node1 and node2
            node1_p = link.left
            node2_p = link.right

            ## set ranks for node1 and node2
            #node1_rank = node1_p.rank # harmful
            #node2_rank = node2_p.rank # harmful
            node1_rank = node1_p.get_rank()
            node2_rank = node2_p.get_rank()

            ## get sizes for node1 and node2
            node1_size = node1_p.get_size()
            node2_size = node2_p.get_size()

            ## set gap sizes for node1 and node2
            #node1_gap_size = node1_p.gap_size # harmful
            #node2_gap_size = node2_p.gap_size # harmful
            node1_gap_size = node1_p.get_gap_size()
            node2_gap_size = node2_p.get_gap_size()

            ## set moment
            node1_moment = math.log(node1_size + 2)/math.log(node1_rank + 2)
            node2_moment = math.log(node2_size + 2)/math.log(node2_rank + 2)

            ## node names
            ## node1, node2 are node names and need to be tuples
            node1, node2  = link.form_paired # assumes a pair of tuples

            ## handle truncated node names
            #node1_alt, node2_alt  = link.form_alt_paired
            ## The code above does not work properly
            node1_alt = node1_p.form_alt
            node2_alt = node2_p.form_alt
            
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
            node1_attrs = NodeAttrs (node1_alt, node1_size, node1_gap_size, node1_rank, node1_zscore, node1_moment)
            node2_attrs = NodeAttrs (node2_alt, node2_size, node2_gap_size, node2_rank, node2_zscore, node2_moment)
            
            ## register node for instances
            if node1_gap_size == 0 and node1_alt not in instances:
                    instances.append (node1_alt)
            if node2_gap_size == 0 and node2_alt not in instances:
                instances.append (node2_alt)

            ## add nodes and edges to G
            ## z-score filtering is moved into draw_graph()
            if not node1 in G.nodes():
                add_node_with_attrs (node1, node1_attrs, G)
            else:
                if check:
                    print(f"#ignored existing node {node1}")
            ## node2
            if not node2 in G.nodes():
                add_node_with_attrs (node2, node2_attrs, G)
            else:
                if check:
                    print(f"#ignored existing node {node2}")

            ### add edges
            if node1 in G.nodes() and node2 in G.nodes():
                G.add_edge (node1, node2)
            if check:
                if node1 not in G.nodes():
                    print(f"#skipped edge: node1 {node1} not found")
                if node2 not in G.nodes():
                    print(f"#skipped edge: node2 {node2} not found")
    ##
    #assert len(instances) > 0 # harmful
    ##
    return G, instances

##
def gen_G_old (N, zscores, zscore_lb, zscore_ub, use_robust_zscore: bool, use_directed_graph: bool, test: bool = True, check: bool = False):
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

    for key, links in sorted (N, reverse = False): # be careful on list up direction
        assert key >= 0
        for link in links:
            if check:
                print(f"#adding link at group {key}: {link}")

            ## get variables
            gap_mark      = link.gap_mark

            ## get patterns for node1 and node2
            node1_p = link.left
            node2_p = link.right

            ## set ranks for node1 and node2
            #node1_rank = node1_p.rank # harmful
            #node2_rank = node2_p.rank # harmful
            node1_rank = node1_p.get_rank()
            node2_rank = node2_p.get_rank()

            ## get sizes for node1 and node2
            node1_size = node1_p.get_size()
            node2_size = node2_p.get_size()

            ## set gap sizes for node1 and node2
            #node1_gap_size = node1_p.gap_size # harmful
            #node2_gap_size = node2_p.gap_size # harmful
            node1_gap_size = node1_p.get_gap_size()
            node2_gap_size = node2_p.get_gap_size()

            ## set moment
            node1_moment = math.log(node1_size + 2)/math.log(node1_rank + 2)
            node2_moment = math.log(node2_size + 2)/math.log(node2_rank + 2)

            ## node names
            ## node1, node2 are node names and need to be tuples
            node1, node2  = link.form_paired # assumes a pair of tuples

            ## handle truncated node names
            #node1_alt, node2_alt  = link.form_alt_paired
            ## The code above does not work properly
            node1_alt = node1_p.form_alt
            node2_alt = node2_p.form_alt
            node1_alt_gap_size = count_items (node1_alt, gap_mark)
            node2_alt_gap_size = count_items (node2_alt, gap_mark)

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
            node1_attrs = NodeAttrs (node1_alt, node1_size, node1_gap_size, node1_rank, node1_moment, node1_zscore)
            node2_attrs = NodeAttrs (node2_alt, node2_size, node2_gap_size, node2_rank, node2_moment, node2_zscore)
            if check:
                if node1 != node1_alt:
                    print(f"#node1_alt: {node1_alt}")
                if node2 != node2_alt:
                    print(f"#node2_alt: {node2_alt}")

            ## register node for instances
            if node1_gap_size == 0 and node1_alt not in instances:
                    instances.append (node1_alt)
            if node2_gap_size == 0 and node2_alt not in instances:
                instances.append (node2_alt)

            ## add nodes and edges to G
            ## case 1: either lowerbound nor upperbound is applied
            if zscore_ub is None and zscore_lb is None:
                ## node1
                if not node1 in G.nodes():
                    add_node_with_attrs (node1, node1_attrs, G)
                else:
                    if check:
                        print(f"#ignored existing node {node1}")
                ## node2
                if not node2 in G.nodes():
                    add_node_with_attrs (node2, node2_attrs, G)
                else:
                    if check:
                        print(f"#ignored existing node {node2}")

            ## when lowerbound and upperbound z-score pruning is applied
            ## case 2: both lowerbound and upperbound
            elif zscore_lb is not None and zscore_ub is not None:
                ## node1
                if node1_zscore >= zscore_lb and node1_zscore <= zscore_ub:
                    if not node1 in G.nodes():
                        add_node_with_attrs (node1, node1_attrs, G)
                    else:
                        if check:
                            print(f"#ignored existing node {node1}")
                else:
                    ## add instance exceptionally
                    if node1_gap_size == 0 and not node1 in G.nodes():
                        add_node_with_attrs (node1, node1_attrs, G)
                    else:
                        print(f"#pruned node {node1}")
                        pruned_node_count += 1
                ## node2
                if node2_zscore >= zscore_lb and node2_zscore <= zscore_ub:
                    if not node2 in G.nodes():
                        add_node_with_attrs (node2, node2_attrs, G)
                    else:
                        if check:
                            print(f"#ignored exisiting node {node2}")
                else:
                    ## add instance exceptionally
                    if node2_gap_size == 0 and not node2 in G.nodes():
                        add_node_with_attrs (node2, node2_attrs, G)
                    else:
                        print(f"#pruned node {node2}")
                        pruned_node_count += 1

            ## case 3: lowerbound only
            elif zscore_lb is not None and zscore_ub is None: # z-score pruning applied
                ## node1
                if node1_zscore >= zscore_lb:
                    if not node1 in G.nodes():
                        add_node_with_attrs (node1, node1_attrs, G)
                    else:
                        if check:
                            print(f"#ignored exisiting node {node1}")
                else:
                    ## add instance exceptionally
                    if node1_gap_size == 0 and not node1 in G.nodes():
                        add_node_with_attrs (node1, node1_attrs, G)
                    else:
                        print(f"#pruned node {node1}")
                        pruned_node_count += 1
                ## node2
                if node2_zscore >= zscore_lb:
                    if not node2 in G.nodes():
                        add_node_with_attrs (node2, node2_attrs, G)
                    else:
                        if check:
                            print(f"#ignored existing node {node2}")
                else:
                    ## add instance exceptionally
                    if node2_gap_size == 0 and not node2 in G.nodes():
                        add_node_with_attrs (node2, node2_attrs, G)
                    else:
                        print(f"#pruned node {node2}")
                        pruned_node_count += 1

            ## case 4: upperbound only
            elif zscore_lb is None and zscore_ub is not None:
                ## node1
                if node1_zscore <= zscore_ub:
                    if not node1 in G.nodes():
                        add_node_with_attrs (node1, node1_attrs, G)
                    else:
                        if check:
                            print(f"ignored existing node {node1}")
                else:
                    print(f"pruned node {node1}")
                    pruned_node_count += 1
                ## node2
                if node2_zscore <= zscore_ub:
                    if not node2 in G.nodes():
                        add_node_with_attrs (node2, node2_attrs, G)
                    else:
                        if check:
                            print(f"ignored existing node {node2}")
                else:
                    ## add instance exceptionally
                    if node2_gap_size == 0 and not node2 in G.nodes():
                        add_node_with_attrs (node2, node2_attrs, G)
                    else:
                        print(f"pruned node {node2}")
                        pruned_node_count += 1

            ## non-existing case
            else:
                raise ValueError("An undefined situation occurred")

            ### add edges
            ## regular form
            if node1 in G.nodes() and node2 in G.nodes():
                G.add_edge (node1, node2)
            ## alt form: unnecessary?
            #if node1_alt in G.nodes() and node2_alt in G.nodes():
            #    G.add_edge (node1_alt, node2_alt)
            if check:
                if node1 not in G.nodes():
                    print(f"#skipped edge: node1 {node1} was not found")
                if node2 not in G.nodes():
                    print(f"#skipped edge: node2 {node2} was not found")
    ##
    #assert len(instances) > 0 # harmful

    ## post-process for z-score pruning
    print(f"#pruned/ignored {pruned_node_count} nodes")

    ##
    return G, instances, pruned_node_count

##
def normalize_zscore (x: float, use_robust_zscore: bool, min_val: float = -5.0, max_val: float = 5.0) -> float:
    "takes a value in the range of min, max and returns its normalized value"
    ##
    import matplotlib.colors as colors
    normalizer = colors.Normalize (vmin = min_val, vmax = max_val)
    return normalizer (x)

##
def get_node_color (node, zscore_dict, use_robust_zscore: bool, normalize_score: bool, check: bool = False):
    """
    generate a value for node color, with or without color normalization.
    """
    try:
        zscore = zscore_dict[node]
        if check:
            print(f"#zscore: {zscore:0.3f}")
        if normalize_score:
            zscore_normalized = normalize_zscore (zscore, use_robust_zscore = use_robust_zscore)
            if check:
                print(f"#zscore_normalized: {zscore_normalized:0.3f}")
            zscore = zscore_normalized
        ##
        node_color = zscore
    except KeyError:
        node_color = 0
    ##
    return node_color

##
def assign_node_colors (G, zscores, instances, use_robust_zscore: bool, normalize_score: bool = False, mark_instances: bool = False, check: bool = False):
    """
    assign colors to nodes using their z-scores
    """
    ## node color setting
    padding_val = 0
    if mark_instances:
        padding_val = 0.0
    ##
    node_colors = []
    for node in G.nodes():
        node_color = get_node_color (node, zscores, use_robust_zscore = use_robust_zscore, normalize_score = normalize_score)
        node_colors.append(node_color)
    ##
    return node_colors

##
def set_node_positions (G, layout: str, MPG_key: str, scale_factor: float):
    """
    set node positions for drawing
    """

    import networkx as nx
    if layout in [ 'Multipartite', 'Multi-partite', 'Multi_partite', 'multi_partite', 'M', 'MP', 'mp' ]:
        layout_name = "Multi-partite"
        ## scale parameter suddenly gets crucial on 2024/10/30
        positions   = nx.multipartite_layout (G, subset_key = MPG_key, scale = -1)
        ## flip x-coordinates when rank is used for MPG_key
        if MPG_key in [ 'rank' ]:
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
        print(f"Unknown layout: Multi-partite (default) is used")
        layout_name = "Multi-partite"
        positions   = nx.multipartite_layout (G, subset_key = MPG_key, scale = -1)
        ##
        if MPG_key in [ 'rank' ]:
            positions = { node: (-x, y) for node, (x, y) in positions.items() }
    ##
    return layout_name, positions

##
def draw_graph (N: dict, layout: str, MPG_key: str = "gap_size", save_instead_of_draw: bool = True, draw_inline: bool = False, auto_figsizing: bool = False, fig_size: tuple = (9,9), fig_dpi: int = 360, node_size: int = None, label_size: int = None, label_sample_n: int = None, zscores: dict = None, p_metric: str = 'rank', use_robust_zscore: bool = True, zscore_lb = None, zscore_ub = None, mark_instances: bool = False, scale_factor: float = 3, generality: int = 0, use_directed_graph: bool = True, reverse_direction: bool = False, font_name: str = None, graphics_backend: str = "qt", check: bool = False) -> None:
    """
    draw a graph from a given network data.
    """

    if check:
        print(f"##N with {len(N)} keys")
        for group_key, links in N:
            print(f"#group {group_key}:\n{links}")

    ##
    import math
    import networkx as nx
    ## select graphics backend
    if not draw_inline:
        import matplotlib
        if graphics_backend is None:
            matplotlib.use('Agg')
        elif graphics_backend == 'qt':
            matplotlib.use('Qt5Agg') # effective
        elif graphics_backend == 'tk':
            matplotlib.use('TkAgg') # default and not effective
        elif graphics_backend == 'gtk':
            matplotlib.use('GTK3Agg') # requires install
        elif graphics_backend == 'wx':
            matplotlib.use('WXAgg') # requires install and not effective
        else:
            matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    from matplotlib import colormaps

    ## generate G
    #G, instances, pruned_node_count = gen_G (N, zscores = zscores, zscore_lb = zscore_lb, zscore_ub = zscore_ub, use_robust_zscore = use_robust_zscore, use_directed_graph = use_directed_graph, check = check)
    G, instances = gen_G (N, zscores = zscores, use_directed_graph = use_directed_graph, check = check)
        ## create a subgraph if needed
    
    ## subgraph creation by z-score filtering
    ## This needs to be applied color assignment
    n_original_nodes = len(G.nodes())
    if zscore_ub is not None and zscore_lb is not None:
        filtered_nodes1 = [ n for n, attr in G.nodes(data = True) if ((attr.get('zscore') >= zscore_lb) and attr.get('zscore') <= zscore_ub)]
        G = G.subgraph (filtered_nodes1).copy()
        #G = G.subgraph (filtered_nodes1) # fails to work
    elif zscore_lb is not None:
        filtered_nodes2 = [ n for n, attr in G.nodes(data = True) if (attr.get('zscore') >= zscore_lb) ]
        G = G.subgraph (filtered_nodes2).copy()
        #G = G.subgraph (filtered_nodes2) # fails to work
    elif zscore_ub is not None:
        filtered_nodes3 = [ n for n, attr in G.nodes(data = True) if (attr.get('zscore') <= zscore_ub) ]
        G = G.subgraph (filtered_nodes3).copy()
        #G = G.subgraph (filtered_nodes3) # fails to work
    
    ## define pruned_node_count
    n_nodes = len(G.nodes())
    pruned_node_count = n_original_nodes - n_nodes

    ## define MPG_key_count_max and MPG_group_size
    MPG_keys = nx.get_node_attributes(G, MPG_key)
    if MPG_keys:
        import collections
        MPG_key_counts = collections.defaultdict(int)
        for k, v in MPG_keys.items():
            MPG_key_counts[v] += 1
        if check:
            print(f"#MPG_key_counts: {MPG_key_counts}")
        MPG_key_count_max = max(MPG_key_counts.values())
        MPG_group_size = len(MPG_key_counts.keys())

    ## color values
    node_colors = assign_node_colors (G, zscores, instances, use_robust_zscore = use_robust_zscore, mark_instances = mark_instances)

    ## relabeling nodes: this needs to come after color setting and before layout setting
    literals = [ x for x in nx.get_node_attributes(G, "literal") ]
    #print(f"literals: {literals}")
    new_labels = { x: as_label(y, sep = " ") for x, y in zip(G, literals) }
    print(f"new_labels: {new_labels}")
    G = nx.relabel_nodes (G, new_labels, copy=True) # copy=False turned out offensive at connectionstyling

    ## set layout and node positions
    layout_name, positions = set_node_positions (G, layout, MPG_key = MPG_key, scale_factor = scale_factor)

    ## set connection
    if layout_name in [ "Multi-partite", "Multi_partite", "Multipartite", "MP" ]:
        connectionstyle = "arc, angleA=0, angleB=180, armA=50, armB=50, rad=15"
    else:
        connectionstyle = "arc"

    ## set figure size
    n_instances = len(instances)
    max_instance_size = 3 # customary
    print(f"#n_instances: {n_instances}")
    if n_instances > 0:
        max_instance_n_segs = max([ len(instance) for instance in instances ])
        print(f"#max_instance_n_segs: {max_instance_n_segs}")
        max_instance_size = max([ sum(map(len, list(instance))) for instance in instances ])
        print(f"#max_instance_size: {max_instance_size}")
    else:
        print(f"#found no genuine instance")

    ## adjust figsize
    if auto_figsizing:
        width_step = max_instance_size * .4
        height_step = 1.00
        if generality in [3]:
            width_step  = round(width_step * 1.0, 1)
            height_step = round(height_step * 2.0, 1)
        elif generality in [1, 2]:
            width_step  = round(width_step * 1.0, 1)
            height_step = round(height_step * 1.5, 1)
        else:
            width_step  = round(width_step * 1.0, 1)
            height_step = round(height_step * 1.5, 1)
        graph_width   = 4 + round(width_step * math.log(1 + MPG_group_size), 1)
        graph_height  = 3 + round(height_step * math.log(1 + MPG_key_count_max), 1)
        #graph_height  = 7 + round(height_step * MPG_key_count_max, 1)
        if graph_width < 4:
            graph_width = 4
        if graph_height < 4:
            graph_height = 4
        fig_size = (graph_width, graph_height)
    print(f"#fig_size: {fig_size}")
    print(f"#fig_dpi: {fig_dpi}")
    #plt.figure(figsize=fig_size, dpi=fig_dpi) # fails to produce right connections in saved file
    plt.figure(figsize=fig_size)

    ## adjust label_size
    resize_coeff = 0.8
    if auto_figsizing:
        label_size = 8 - round(resize_coeff * math.log(1 + n_instances), 1)
    print(f"#label_size: {label_size}")

    ## adjust node_size
    if auto_figsizing:
        node_size = 8 - round(resize_coeff * math.log(1 + n_instances), 1)
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
        min_source_margin = 10,  # These work here
        min_target_margin = 10
    )

    ## Create custom label positions with offset
    label_offset_x = 0.002  # Adjust these values as needed
    label_offset_y = 0.005
    label_positions = {
        node: (x + label_offset_x, y + label_offset_y)
        for node, (x, y) in positions.items()
    }

    ## Draw labels separately with full control
    nx.draw_networkx_labels(G, label_positions,
        font_family = font_family,
        font_color = 'darkblue',
        font_size = label_size,
        verticalalignment = "top",  # or "top", "center"
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

    ### set title
    pl_type = f"g{generality}PL"
    if layout_name in ['Multi-partite']:
        layout_name = f"{layout_name} [key: {MPG_key}]"
    if use_robust_zscore:
        title_val = f"{pl_type} (layout: {layout_name}; robust z-scores [p_metric: {p_metric}]: {zscore_lb} – {zscore_ub} [removed {pruned_node_count} nodes]) built from\n{instance_labels} ({label_count} in all)"
    else:
        title_val = f"{pl_type} (layout: {layout_name}; normal z-scores [p_metric: {p_metric}]: {zscore_lb} – {zscore_ub} [removed {pruned_node_count} nodes]) built from\n{instance_labels} ({label_count} in all)"
    plt.title(title_val)
    ##
    #plt.tight_layout()
    #plt.set_dpi(fig_dpi) # fails
    if save_instead_of_draw:
        file_name = f"{pl_type}.png"
        #plt.savefig(file_name, dpi = fig_dpi, bbox_inches = 'tight')
        #plt.savefig(file_name, dpi = fig_dpi)
        ## dip = value above turned out to be offensive!
        plt.savefig(file_name)
        print(f"#saved figure as <{file_name}> in the current directory")
    else:
        plt.show()


## Classes
class PatternLattice():
    """
    definition of PatternLattice class
    """

    ##
    def __init__ (self, pattern, generality: int, p_metric: str = 'rank', reflexive: bool = True, reductive: bool = True, make_links_safely: bool = False, check: bool = False):
        """
        initialization of a PatternLattice, or PL
        """

        if check:
            print(f"pattern.paired: {pattern.paired}")
        ##
        self.origin           = pattern
        self.generality       = generality
        self.p_metric         = p_metric
        self.nodes            = pattern.build_lattice_nodes (generality = generality, check = check)
        self.gap_mark         = self.nodes[0].gap_mark
        ## gen_links, and instantiation_check assume the following
        #self.ranked_nodes     = self.get_nodes_grouped_by_rank (check = check)
        self.ranked_nodes     = self.get_nodes_grouped_by ('rank', check = check)
        ## gen_link() needs to be rank-based
        self.links            = self.gen_links (reflexive = reflexive, check = check)
        self.link_sources, self.link_targets = self.get_link_stats (check = check)
        #self.ranked_links     = make_links_ranked (self.links, safely = make_links_safely, check = check)
        self.ranked_links     = make_links_grouped_by ('rank', self.links, use_max = False, safely = make_links_safely, check = check)
        self.gap_sized_links    = make_links_grouped_by ('gap_size', self.links, use_max = False, safely = make_links_safely, check = check)
        ## z-scores
        self.source_zscores  = {}
        self.target_zscores  = {}
        ## aliases
        self.zscores_from_sources = self.source_zscores
        self.zscores_from_targets = self.target_zscores
        ## robust z-scores
        self.source_robust_zscores  = {}
        self.target_robust_zscores  = {}
        ## aliases
        self.robust_zscores_from_sources = self.source_robust_zscores
        self.robust_zscores_from_targets = self.target_robust_zscores

    ##
    def __len__(self):
        """
        define response to len(...)
        """
        return (len(self.nodes), len(self.links))

    ##
    def __repr__(self):
        """
        define response to print(...)
        """
        return f"{type(self).__name__} ({self.nodes!r})"

    ##
    def __iter__(self):
        """
        define response to iter(...)
        """
        #for x in self.nodes: yield x
        ## the above is replaced by the following
        return iter (self)

    ##
    def print (self):
        """
        defines response to print(...)
        """

        out = f"{type(self).__name__} ({self.nodes!r})\n"
        out += f"{type(self).__name__} ({self.source_zscores!r})\n"
        return out

    ##
    def merge_with (self, other, **params):
        """
        take two PatternLattices and merge them into one.
        """
        gen_links_internally = params['gen_links_internally']
        generality           = params['generality']
        reflexive            = params['reflexive']
        reductive            = params['reductive']
        use_mp               = params['use_mp']
        check                = params['check']

        ## merger nodes of two pattern lattices given
        main_nodes   = [ p for p in self.nodes if len(p) > 0 ]
        other_nodes = [ p for p in other.nodes if len(p) > 0 ]

        ## variables
        #gap_mark = main_nodes[0].gap_mark
        #tracer   = main_nodesd[0].tracer
        ## The code above fails when self is a null Pattern Lattice
        gap_mark = other_nodes[0].gap_mark
        tracer   = other_nodes[0].tracer
        p_metric = other.p_metric

        ##
        if reductive:
            pool_nodes  = simplify_list (main_nodes)
            other_nodes = simplify_list (other_nodes)
        ##
        if reductive:
            try:
                main_set = set(main_nodes)
                other_set = set(other_nodes)
                main_nodes = list(main_set | other_set)
            except TypeError:
                main_nodes = make_simplest_merger(main_nodes, other_nodes)
        else:
            main_nodes = make_simplest_merger (main_nodes, other_nodes)
        if check:
            for i, node in enumerate(main_nodes):
                print(f"#main_node {i}: {node.separated_print()}")

        ## define a new pattern lattice and elaborates it
        dummy_pattern = Pattern([], gap_mark = gap_mark, tracer = tracer, check = check)
        merged = PatternLattice (dummy_pattern, generality = generality, p_metric = p_metric, reductive = reductive, check = check)
        ##
        merged.origin        = dummy_pattern
        merged.nodes         = main_nodes
        ## The following was a seriously elusive bug
        #merged.ranked_nodes  = group_nodes_by_rank (merged.nodes, gap_mark = gap_mark)
        merged.ranked_nodes  = merged.get_nodes_grouped_by_rank (check = check)
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
    def get_nodes_grouped_by (self, metric: str, check: bool = False) -> dict:
        """
        takes a list of patterns, P, and generates a dictionary of patterns grouped by p_metric
        """

        gap_mark   = self.gap_mark
        nodes      = self.nodes
        size       = len(nodes)

        ## implementation using itertools.groupby() failed
        if metric == 'gap_size':
            metric_finder = lambda p: len([ x for x in p.form if len(x) > 0 and x == gap_mark ])
        elif metric == 'rank':
            metric_finder = lambda p: len([ x for x in p.form if len(x) > 0 and x != gap_mark ])
        else:
            raise ValueError("Undefined p_metric")

        ## main
        from collections import defaultdict
        grouped_nodes = defaultdict(list) # dictionary
        for pattern in sorted (nodes, key = metric_finder):
            if metric == 'gap_size':
                metric_val = pattern.get_gap_size ()
            elif metric == 'rank':
                metric_val = pattern.get_rank ()
            if check:
                print(f"#rank: {pattern_rank}")
                print(f"#ranked pattern: {pattern}")
            if metric_val <= size:
                grouped_nodes[metric_val].append(pattern)

        ## check and return
        if check:
            print(f"#grouped_nodes: {grouped_nodes}")
        return grouped_nodes

    ##
    def get_nodes_grouped_by_rank (self, check: bool = False) -> dict:
        """
        takes a list of patterns, P, and generates a dictionary of patterns grouped by their ranks
        """

        from collections import defaultdict
        gap_mark   = self.gap_mark
        nodes      = self.nodes
        size       = len(nodes)

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

    ## generate links: instantiation check not accept gap_size
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

        ##
        increment = 1
        for rank in sorted (ranks, reverse = False):
            selected_links = [ ]
            try:
                L = simplify_list (ranked_nodes[rank])
                if check:
                    print(f"#L rank {rank} nodes: {L}")
                R = simplify_list (ranked_nodes[rank + increment])
                if check:
                    print(f"#R rank {rank + increment} nodes: {R}")
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
                    selected_links = classify_relations_nmp (R, L, check = check)
                links.extend (selected_links)
            except KeyError:
               pass
        ##
        return links

    ## update_links() needs to be rank-based
    def update_links (self, reflexive: bool, use_mp: bool = False, make_links_safely: bool = False, check: bool = False):
        """
        takes a PatternLattice P, and updates P.links, P.link_sources and P.link_targets.
        """
        ## update links
        self.links  = self.gen_links (reflexive = reflexive, use_mp = use_mp, check = check)
        ## update ranked_links: metric needs to be 'rank'
        self.ranked_links  = make_links_grouped_by ('rank', self.links, use_max = False, safely = make_links_safely, check = check)
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
                link_targets[r_form] += 1
                seen.append(link)
        ## return result
        return link_sources, link_targets

    ##
    def draw_lattice (self, layout: str = None, MPG_key: str = None, save_instead_of_draw: bool = True, draw_inline: bool = False, auto_figsizing: bool = False, fig_size: tuple = None, fig_dpi: int = 620, generality: int = 0, p_metric: str = 'rank', make_links_safely: bool = False, use_robust_zscore: bool = True, zscores_from_targets: bool = False, zscore_lb: float = None, zscore_ub: float = None, mark_instances: bool = False, node_size: int = 11, label_size: int = 9, label_sample_n: int = None, scale_factor: float = 3, graphics_backend: str = 'qt', font_name: str = None, check: bool = False) -> None:
        """
        draws a lattice digrams from a given PatternLattice L by extracting L.links
        """
        ##
        links  = self.links
        if check:
            print(f"#links: {links}")
        ##
        sample_pattern = self.nodes[0]
        gap_mark       = sample_pattern.gap_mark
        ranked_links  = make_links_grouped_by ('rank', links, use_max = False, safely = make_links_safely, check = check)
        if check:
            for rank, links in ranked_links.items():
                print(f"#links at rank {rank}:\n{links}")

        ## handle z-scores
        if zscores_from_targets:
            if use_robust_zscore:
                zscores = self.target_robust_zscores
            else:
                zscores = self.target_zscores
        else:
            if use_robust_zscore:
                zscores = self.source_robust_zscores
            else:
                zscores = self.source_zscores
        ##
        if check:
            for i, item in enumerate(zscores.items()):
                node, v = item
                print(f"node {i:4d} {node} has z-score {v:.4f}")

        ## draw PatternLattice
        draw_graph (ranked_links.items(), layout = layout, MPG_key = MPG_key, save_instead_of_draw = save_instead_of_draw, draw_inline = draw_inline, auto_figsizing = auto_figsizing, fig_size = fig_size, fig_dpi = fig_dpi, node_size = node_size, label_size = label_size, generality = generality, scale_factor = scale_factor, label_sample_n = label_sample_n, graphics_backend = graphics_backend, font_name = font_name, p_metric = p_metric, zscores = zscores, use_robust_zscore = use_robust_zscore, zscore_lb = zscore_lb, zscore_ub = zscore_ub, mark_instances = mark_instances, check = check)

### end of file
