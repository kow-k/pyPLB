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

### Functions

def test_for_ISA_relation (l: list, r: list, check: bool = False) -> list:
    '''
    tests if a given pair of Patterns is in IS-A relation
    '''
    gap_mark = r.gap_mark
    r_form = r.form
    r_size = len(r_form)
    r_rank = get_rank_of_list (r_form, gap_mark)
    l_form = l.form
    l_size = len(l_form)
    l_rank = get_rank_of_list (l_form, gap_mark)
    ##
    if abs(l_size - r_size) > 1:
        if check:
            print(f"#is-a:F0; {l.form} ~ {r.form}")
        #continue
        return False
    ##
    elif l_size == r_size + 1:
        if l_form[:-1] == r_form or l_form[1:] == r_form:
            print(f"#is-a:T1; {l.form} <- {r.form}")
            return True
        else:
            if check:
                print(f"#is-a:F1; {l.form} <- {r.form}")
            return False
    ##
    elif l_size == r_size:
        if r.form == l.form:
            if check:
                print(f"#is-a:F2; {l.form} ~ {r.form}")
            return False
        if r.count_gaps() == 0 and l.count_gaps() == 1:
            if r.includes(l):
                print(f"#is-a:T0:instance' {l.form} <- {r.form}")
                return True
            else:
                if check:
                    print(f"#is-a:F3; {l.form} ~ {r.form}")
                return False
        elif check_for_instantiation (r, l, check = False):
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
def classify_relations_mp (R, L, check: bool = False):
    """
    takes two Patterns, classify their relation and returns the list of is-a cases.

    in logging, "is-a:Ti" means is-a relation is True of case i; "is-a:Fi" means is-a relation is False of case i;
    """
    ##
    #if len(R) == 0 or len(L) == 0:
    #    return []
    ## generates a list of Boolean values
    try:
        gap_mark = R[0].left.gap_mark
    except IndexError:
        print(f"#R: {R}")
        try:
            gap_mark = L[0].left.gap_mark
        except IndexError:
            print(f"#L: {L}")
    ##
    from itertools import product
    R2 = sorted (R, key = lambda x: len(x), reverse = False)
    L2 = sorted (L, key = lambda x: len(x), reverse = False)
    #pairs = product (R2, L2) # cannot be reused
    import multiprocess as mp
    import os
    pool = mp.Pool(max(os.cpu_count(), 1))
    test_values = pool.starmap (test_for_ISA_relation, product (R2, L2))
    print (f"test_values: {test_values}")
    ## remove duplicates
    #seen  = [ ]
    #links = [ ]
    #for link in [ link for link, value in zip ([ PatternLink ((r, l)) for r, l in product (R2, L2) ], test_values) if value is True ]:
    #    if not link in seen:
    #        links.append(link)
    raw_links = [ link for link, value in zip ([ PatternLink ((r, l)) for r, l in product (R2, L2) ], test_values) if value is True ]
    print(f"raw_links: {raw_links}")
    null_pat = Pattern ([], gap_mark = gap_mark)
    links = sort_remove_duplicates (raw_links, PatternLink ((null_pat, null_pat)))
    #print (links)
    ##
    return links

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
            r_form = r.form
            r_size = len(r_form)
            r_rank = get_rank_of_list (r_form, gap_mark)
            l_form = l.form
            l_size = len(l_form)
            l_rank = get_rank_of_list (l_form, gap_mark)
            ##
            if abs(l_size - r_size) > 1:
                if check:
                    print(f"#is-a:F0; {l.form} ~ {r.form}")
                continue
            ##
            elif l_size == r_size + 1:
                if l_form[:-1] == r_form or l_form[1:] == r_form:
                    print(f"#is-a:T1; {l.form} <- {r.form}")
                    link = PatternLink ((l, r))
                    register_link (link)
                else:
                    if check:
                        print(f"#is-a:F1; {l.form} <- {r.form}")
                    continue
            ##
            elif l_size == r_size:
                if r.form == l.form:
                    if check:
                        print(f"#is-a:F2; {l.form} ~ {r.form}")
                    continue
                if r.count_gaps() == 0 and l.count_gaps() == 1:
                    if r.includes(l):
                        print(f"#is-a:T0:instance' {l.form} <- {r.form}")
                        link = PatternLink ((l, r))
                        register_link (link)
                    else:
                        if check:
                            print(f"#is-a:F3; {l.form} ~ {r.form}")
                        continue
                elif check_for_instantiation (r, l, check = False):
                    print(f"#is-a:T2; {l.form} <- {r.form}")
                    link = PatternLink ((l, r))
                    register_link (link)
                else:
                    if check:
                        print(f"#is-a:F3; {l.form} ~ {r.form}")
                    continue
            ##
            else:
                if check:
                    print(f"#is-a:F4; {l.form} ~ {r.form}")
                continue
    ##
    return sub_links


##
def draw_network (D: dict, layout: str, fig_size: tuple = None, auto_fig_sizing: bool = False, label_size: int = None, label_sample_n: int = None, node_size: int = None, zscores: dict = None, use_robust_zscore: bool = False, zscore_lb = None, zscore_ub = None, scale_factor: float = 3, font_name: str = None, generalized: bool = True, test: bool = False, use_pyGraphviz: bool = False, use_directed_graph: bool = True, reverse_direction: bool = False, check: bool = False) -> None:
    "draw layered graph under multi-partite setting"
    ##
    import networkx as nx
    import math
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import seaborn as sns

    ## define graph
    if use_directed_graph:
        G = nx.DiGraph()
    else:
        G = nx.Graph() # does not accept connectionstyle specification
    ##
    try:
        rank_max = max(int(x[0]) for x in list(D))
    except ValueError:
        rank_max = 3
    ##
    node_dict = { }
    instances = [ ] # register instances
    node_counts_by_layers = [ ]
    pruned_node_count = 0
    for rank, links in sorted (D, reverse = True): # be careful on list up direction
        L, R, E = [], [], []
        for link in links:
            if check:
                print(f"#adding link at rank {rank}: {link}")
            ## process nodes
            gap_mark      = link.gap_mark
            node1, node2  = link.form_paired
            node1_rank    = get_rank_of_list (node1, gap_mark)
            node2_rank    = get_rank_of_list (node2, gap_mark)

            ## register node for instances
            if count_items (node1, gap_mark) == 0 and node1 not in instances:
                instances.append (node1)
            if count_items (node2, gap_mark) == 0 and node2 not in instances:
                instances.append (node2)

            ## assign z-scores
            try:
                node1_zscore = zscores[node1]
            except KeyError:
                node1_zscore = 0
            try:
                node2_zscore = zscores[node2]
            except KeyError:
                node2_zscore = 0

            ## add nodes
            ## when lowerbound and upperbound z-score pruning is applied
            if zscore_ub is not None and zscore_lb is not None: # z-score pruning applied
                ## node1
                if node1_zscore >= zscore_lb and node1_zscore <= zscore_ub and node1_rank == rank and not node1 in L:
                    L.append (node1)
                else:
                    print(f"pruned node {node1} with z-score {node1_zscore: 0.4f}")
                    pruned_node_count += 1
                ## node2
                if node2_zscore >= zscore_lb and node2_zscore <= zscore_ub:
                    if node2_rank == rank + 1 and not node2 in R:
                        R.append (node2)
                    elif not node2 in L:
                        L.append (node2)
                else:
                    print(f"pruned node {node2} with z-score {node2_zscore: 0.4f}")
                    pruned_node_count += 1
                ## process edges
                if node1_zscore >= zscore_lb and node1_zscore <= zscore_ub and node2_zscore >= zscore_lb and node2_zscore <= zscore_ub:
                    edge = (node1, node2)
                    #edge = (node2, node1)
                try:
                    if edge and not edge in E:
                        E.append (edge)
                except UnboundLocalError:
                    pass
            ## when upperbound z-score pruning is applied
            elif not zscore_ub is None: # z-score pruning applied
                ## node1
                if node1_zscore <= zscore_ub and not node1 in L:
                    L.append (node1)
                else:
                    print(f"pruned node {node1} with z-score {node1_zscore: 0.4f}")
                    pruned_node_count += 1
                ## node2
                if node2_zscore <= zscore_ub and get_rank_of_list (node2, gap_mark) == rank and not node2 in R:
                    R.append (node2)
                elif node2_zscore <= zscore_ub and not node2 in L:
                    R.append (node2)
                else:
                    print(f"pruned node {node2} with z-score {node2_zscore: 0.4f}")
                    pruned_node_count += 1
                ## register instance nodes
                if count_items (node2, gap_mark) == 0 and node2 not in instances:
                    instances.append (node2)
                ## process edges
                if node1_zscore <= zscore_ub and node2_zscore <= zscore_ub:
                    edge = (node1, node2)
                    #edge = (node2, node1)
                try:
                    if edge and not edge in E:
                        E.append (edge)
                except UnboundLocalError:
                    pass
            ## when lowerbound z-score pruning is applied
            elif not zscore_lb is None: # z-score pruning applied
                ## node1
                if node1_zscore >= zscore_lb and not node1 in L:
                    L.append (node1)
                else:
                    print(f"pruned node {node1} with z-score {node1_zscore: 0.4f}")
                    pruned_node_count += 1
                ## node2
                if node2_zscore >= zscore_lb and get_rank_of_list (node2, gap_mark) == rank and not node2 in R:
                    R.append (node2)
                elif node2_zscore >= zscore_lb and not node2 in L:
                    R.append (node2)
                else:
                    print(f"pruned node {node2} with z-score {node2_zscore: 0.4f}")
                    pruned_node_count += 1
                ## process edges
                if node1_zscore >= zscore_lb and node2_zscore >= zscore_lb:
                    edge = (node1, node2)
                    #edge = (node2, node1)
                try:
                    if edge and not edge in E:
                        E.append (edge)
                except UnboundLocalError:
                    pass
            ## when z-score pruning not applied
            else:
                ## node1
                if node1_rank == rank and not node1 in L:
                    L.append (node1)
                ## node2
                if node2_rank == rank + 1:
                    if not node2 in R:
                        R.append (node2)
                elif node2_rank == rank and not node2 in L:
                    L.append (node2)
                ## process edges
                if node1 and node2:
                    edge = (node1, node2)
                    #edge = (node2, node1)
                if edge and not edge in E:
                    E.append (edge)

        ## populates nodes for G
        ## forward rank scan = rank increments
        #G.add_nodes_from (L, rank = rank)
        #G.add_nodes_from (R, rank = rank + 1)
        ## backward rank scan = rank decrements
        G.add_nodes_from (R, rank = (rank_max - rank - 1))
        G.add_nodes_from (L, rank = (rank_max - rank))

        ## populates edges for G
        G.add_edges_from (E)

        ## update node_counts_by_layers
        node_counts_by_layers.append (len(R))

    ## post-process for z-score pruning
    print(f"#pruned {pruned_node_count} nodes")

    ## post-process for max_node_count_by_layers
    try:
        max_node_count_on_layer = max(node_counts_by_layers)
    except ValueError:
        max_node_count_on_layer = 4

    ## node color setting
    values_for_color = []
    for node in G:
        try:
            z_value = zscores[node]
            if check:
                print(f"#z_value: {z_value: 0.4f}")
            z_normalized = normalize_zscore(z_value, use_robust_zscore = use_robust_zscore)
            if check:
                print(f"#z_normalized: {z_normalized: 0.4f}")
            values_for_color.append (z_normalized)
        except KeyError:
            values_for_color.append (0.5) # normalized value falls between 0 and 1.0

    ## relabeling nodes: this needs to come after color setting
    new_labels = { x: as_label(x, sep = " ", add_sep_at_end = True) for x in G }
    G = nx.relabel_nodes (G, new_labels, copy = False)

    ## set positions
    if use_pyGraphviz:
        nx.nx_agraph.view_pygraphviz(G, prog = 'fdp')
    else:
        ## select layout
        if layout in [ 'Multipartite', 'Multi_partite', 'multi_partite', 'M', 'MP', 'mp' ]:
            layout_name = "Multi-partite"
            ## scale parameter suddenly gets crucial on 2024/10/30
            positions   = nx.multipartite_layout (G, subset_key = "rank", scale = -1)
        ##
        elif layout in [ 'Graphviz', 'graphviz', 'G' ] :
            layout_name = "Graphviz"
            positions   = nx.nx_pydot.graphviz_layout(G) # obsolete?
            #positions = nx.nx_agraph.graphviz_layout(G)
        ##
        elif layout in ['arf', 'ARF' ] :
            layout_name = "ARF"
            positions   = nx.arf_layout(G, scaling = scale_factor)
        ##
        elif layout in [ 'Fruchterman-Reingold', 'Fruchterman_Reingold', 'fruchterman_reingold', 'FR']:
            layout_name = "Fruchterman-Reingold"
            positions   = nx.fruchterman_reingold_layout (G, scale = scale_factor, dim = 2)
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
            positions   = nx.multipartite_layout (G, subset_key = "rank", scale = -1)

    ### draw
    ## set connection
    if layout_name == "Multi-partite":
        connectionstyle = "arc, angleA=0, angleB=180, armA=50, armB=50, rad=15"
    else:
        connectionstyle = "arc"

    ## set figure size
    if not fig_size is None:
        fig_size_local = fig_size
    else:
        if auto_fig_sizing:
            fig_size_local = \
                (round(2 * len(D), 0), round(2 * math.log (max_node_count_on_layer), 0))
        else:
            pass
    try:
        print(f"#fig_size_local: {fig_size_local}")
        plt.figure(figsize = fig_size_local)
    except NameError:
        pass

    ## set font_size
    if auto_fig_sizing:
        if label_size is None:
            try:
                font_size = round(label_size/1.5 * math.log (max_node_count_on_layer), 0)
            except (ZeroDivisionError, TypeError):
                font_size = 7
    else:
        if not label_size is None:
            font_size = label_size
        else:
            font_size = 7
    print(f"#font_size: {font_size}")

    ## set node_size
    if node_size is None:
        node_size = 12
    else:
        try:
            node_size = round (1.2 * node_size/math.log (max_node_count_on_layer), 0)
        except ZeroDivisionError:
            node_size = 12
    print(f"#node_size: {node_size}")

    ## set font name
    if font_name is None:
        font_family = "Sans-serif"
    else:
        font_family = font_name

    ## set colormap
    my_cmap = sns.color_palette("coolwarm", 24, as_cmap = True) # Crucially, as_cmap

    ## revserse the arrows
    if use_directed_graph and reverse_direction:
        G = G.reverse(copy = False) # offensive?

    ## finally draw
    nx.draw_networkx (G, positions,
        font_family = font_family,
        font_color = 'darkblue', # label font color
        verticalalignment = "bottom", horizontalalignment = "right",
        min_source_margin = 6, min_target_margin = 6,
        font_size = font_size, node_size = node_size,
        node_color = values_for_color, cmap = my_cmap,
        edge_color = 'gray', width = 0.1, arrowsize = 6,
        arrows = True, connectionstyle = connectionstyle,
    )

    ## set labels used in title
    #instance_labels = [ as_label (x, sep = ",") for x in sorted (instances) ]
    instance_labels = [ as_label (x, sep = ",") for x in instances ]
    label_count = len (instance_labels)
    if label_sample_n is not None and label_count > label_sample_n:
        new_instance_labels = instance_labels[:label_sample_n - 1]
        new_instance_labels.append("…")
        new_instance_labels.append(instance_labels[-1])
        instance_labels = new_instance_labels
    print(f"#instance_labels {label_count}: {instance_labels}")

    ### set title
    if generalized:
        if use_robust_zscore:
            title_val = f"gPatternLattice (layout: {layout_name}; robust z-scores: {zscore_lb} – {zscore_ub}) built from\n{instance_labels} ({label_count} in all)"
        else:
            title_val = f"gPatternLattice (layout: {layout_name}; normal z-scores: {zscore_lb} – {zscore_ub}) built from\n{instance_labels} ({label_count} in all)"
    else:
        if use_robust_zscore:
            title_val = f"PatternLattice (layout: {layout_name}; robust z-scores: {zscore_lb} – {zscore_ub}) built from\n{instance_labels} ({label_count} in all)"
        else:
            title_val = f"PatternLattice (layout: {layout_name}; normal z-scores: {zscore_lb} – {zscore_ub}) built from\n{instance_labels} ({label_count} in all)"
    plt.title(title_val)
    ##
    plt.show()

##
def make_ranked_dict (L: list, gap_mark: str) -> dict:
    "takes a list of lists and returns a dict whose keys are ranks of the lists"
    ##
    ranked_dict = {}
    for rank in set([ get_rank_of_list (x, gap_mark) for x in L ]):
        ranked_dict[rank] = [ x for x in L if Pattern(x, gap_mark).get_rank() == rank ]
    ##
    return ranked_dict

## alises
#group_nodes_by_rank = make_ranked_dict # Turned out truly offensive!

#make_links_ranked   = make_ranked_dict
## The following is needed independently of make_ranked_dict(..)
def make_links_ranked (L: list, check: bool = False) -> list:
    "takes a list of PatternLinks and returns a dictionary of {rank: [link1, link2, ...]}"
    ranked_links = {}
    for link in L:
        rank = link.get_link_rank ()
        try:
            if not link in ranked_links[rank]:
                ranked_links[rank].append(link)
        except KeyError:
            ranked_links[rank] = [link]
    ##
    return ranked_links

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
        #print(f"#members: {members}")
        stats['n_members'] = len(members)
        #print(f"#n_members: {n_members}")
        dist = [ link_dict[m] for m in members ]
        #print(f"dist: {dist}")
        stats['dist'] = dist
        ##
        rank_dists[rank] = stats
    ##
    return rank_dists

##
def merge_patterns_and_filter (A, B, check = False):
    #return A.merge_patterns (B, check = check) # turned out to be offensive
    C = A.merge_patterns (B, check = check)
    #if C is not None: # fails to work
    if len(C) > 0:
        return C
    #    #yield C # fails

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
def gen_zscores_from_sources (M, gap_mark: str, use_robust_zscore: bool, check: bool = False):
    ## adding link source z-scores to M
    Link_sources     = M.link_sources
    if check:
        print(f"##Link_sources")
    ranked_links     = make_ranked_dict (Link_sources, gap_mark = gap_mark)
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

def gen_zscores_from_targets (M, gap_mark: str, use_robust_zscore: bool, check: bool = False):
    ## adding link target z-scores to M
    Link_targets     = M.link_targets
    if check:
        print(f"##Link_targets")
    ranked_links     = make_ranked_dict (Link_targets, gap_mark = gap_mark)
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

## Classes
##
class PatternLattice():
    "definition of PatternLattice class"
    ##
    def __init__ (self, pattern, generalized: bool, reflexive: bool = True, reductive: bool = True, check: bool = False):
        "initialization of a PatternLattice"
        if check:
            print(f"pattern.paired: {pattern.paired}")
        ##
        self.origin       = pattern
        self.generalized  = generalized
        self.nodes        = pattern.build_lattice_nodes (generalized = generalized, check = check)
        self.gap_mark     = self.nodes[0].gap_mark
        self.ranked_nodes = self.group_nodes_by_rank (check = check)
        ## old code
        #self.links, self.link_sources, self.link_targets = self.gen_links (reflexive = reflexive, check = check)
        self.links          = self.gen_links (reflexive = reflexive, check = check)
        self.ranked_links   = make_links_ranked (self.links, check = check)
        self.link_sources, self.link_targets = self.get_link_stats (check = check)
        self.source_zscores = {}
        self.target_zscores = {}
        #return self # This may not be run

    ##
    def __len__(self):
        return (len(self.nodes), len(self.links))

    ##
    def __repr__(self):
        return f"{type(self).__name__} ({self.nodes!r})"

    ##
    def __iter__(self):
        for x in self.nodes:
            yield x

    ##
    def print (self):
        out = f"{type(self).__name__} ({self.nodes!r})\n"
        out += f"{type(self).__name__} ({self.source_zscores!r})\n"
        return out

    ##
    def group_nodes_by_rank (self, check: bool = False) -> dict:
        "takes a list of patterns, P, and generates a dictionary of patterns grouped by their ranks"
        ##
        from collections import defaultdict
        gap_mark  = self.gap_mark
        nodes     = self.nodes
        size      = len(nodes)
        ## implementation using itertooks.groupby() failed
        rank_finder = lambda p: len([ x for x in p.form if x != gap_mark ])
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
    def gen_links (self, reflexive: bool = True, use_mp: bool = False, check: bool = False):
        """
        takes a PatternLattice P, and generates data for for P.links
        """
        ##
        ranked_nodes = self.ranked_nodes
        if len (ranked_nodes) == 0:
            ranked_nodes = make_ranked_dict (self.nodes)
        ##
        links =  [ ]
        ranks = ranked_nodes.keys()
        for rank in sorted (ranks, reverse = False):
            selected_links = [ ]
            try:
                L = simplify_list (ranked_nodes[rank])
                if check:
                    print(f"#L rank {rank} nodes: {L}")
                R = simplify_list (ranked_nodes[rank + 1] )
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
                #selected_links = classify_relations_mp (R, L, check = check)
                selected_links = classify_relations (R, L, check = check)
                links.extend (selected_links)
            except KeyError:
               pass
        ##
        #supplement_links = list((gap_mark,)*i for i in range(max(ranks) + 1))
        #return links + supplement_links
        return links

    ##
    def update_links (self, reflexive: bool, use_mp: bool = False, check: bool = False):
        """
        takes a PatternLattice P, and updates P.links, P.link_sources and P.link_targets.
        """
        ## update links
        self.links  = self.gen_links (reflexive = reflexive, use_mp = use_mp, check = check)
        ## update ranked_links
        self.ranked_links  = make_links_ranked (self.links, check = check)
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
        ##
        link_sources, link_targets = defaultdict(int), defaultdict(int)
        seen = [ ]
        for link in sorted (self.links, key = lambda x: len(x), reverse = False):
            if not link in seen:
                l_form, r_form = link.left.form, link.right.form
                link_sources[l_form] += 1
                #if link.right.count_gaps() > 0:
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
        reductive            = params['reductive']
        reflexive            = params['reflexive']
        use_mp               = params['use_mp']
        check                = params['check']

        ## merger nodes of two pattern lattices given
        main_nodes   = [ p for p in self.nodes if len(p) > 0 ]
        nodes_to_add = [ p for p in other.nodes if len(p) > 0 ]
        ##
        gap_mark = main_nodes[0].gap_mark
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
        dummy_pattern = Pattern([], gap_mark = gap_mark, check = check)
        merged = PatternLattice (dummy_pattern, generalized = generalized, reductive = reductive, check = check)
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
    def draw_diagrams (self, generalized: bool, zscores_from_targets: bool, layout: str = None, auto_fig_sizing: bool = False, zscore_lb: float = None, zscore_ub: float = None, use_robust_zscore: bool = False, scale_factor: float = 3, fig_size: tuple = None, label_size: int = None, label_sample_n: int = None, node_size: int = None, font_name: str = None, use_pyGraphviz: bool = False, test: bool = False, check: bool = False) -> None:
        """
        draw a lattice digrams from a given PatternLattice L by extracting L.links
        """
        ##
        #generalized = self.generalized
        links       = self.links
        if check:
            print(f"#links: {links}")
        ##
        sample_pattern = self.nodes[0]
        gap_mark = sample_pattern.gap_mark
        ranked_links = make_links_ranked (links)
        if check:
            for rank, links in ranked_links.items():
                print(f"#links at rank {rank}:\n{links}")

        ## handle z-scores
        #zscores = self.source_zscores
        if zscores_from_targets:
            zscores = self.target_zscores
            #zscores = self.source_zscores
        else:
            zscores = self.source_zscores
        ##
        if check:
            i = 0
            for node, v in zscores.items():
                i += 1
                print(f"node {i:4d} {node} has z-score {v:.4f}")

        ## draw PatternLattice
        draw_network (ranked_links.items(), generalized = generalized, layout = layout, fig_size = fig_size, auto_fig_sizing = auto_fig_sizing, node_size = node_size, scale_factor = scale_factor, label_sample_n = label_sample_n, font_name = font_name, zscores = zscores, use_robust_zscore = use_robust_zscore, zscore_lb = zscore_lb, zscore_ub = zscore_ub, check = check)


### end of file
