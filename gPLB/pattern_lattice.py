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

## additional imports
#from numba import jit # incompatible with PatternLattice class
#import itertools
#import numpy as np
#import scipy.stats as stats
#import multiprocessing as mp
from collections import defaultdict

### Functions

##
def draw_network (D: dict, layout: str, fig_size: tuple = None, auto_fig_sizing: bool = False, label_size: int = None, label_sample_n: int = None, node_size: int = None, zscores: dict = None, use_robust_zscore: bool = False, zscore_lowerbound = None, zscore_upperbound = None, scale_factor: float = 3, font_name: str = None, generalized: bool = True, test: bool = False, use_pyGraphviz: bool = False, use_directed_graph: bool = True, reverse_direction: bool = False, check: bool = False) -> None:
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
    node_dict = { }
    instances = [ ] # register instances
    node_counts_by_layers = [ ]
    ##
    rank_max = max(int(x[0]) for x in list(D))
    pruned_node_count = 0
    for rank, links in sorted (D, reverse = True): # be careful on list up direction
        L = [ ]
        R, E = [ ], [ ]
        for link in links:
            if check:
                print(f"#adding link at rank {rank}: {link}")

            ## process nodes
            gap_mark      = link.gap_mark
            node1, node2  = link.form_paired

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
            if zscore_upperbound is not None and zscore_lowerbound is not None: # z-score pruning applied
                ## node1
                if node1_zscore >= zscore_lowerbound and node1_zscore <= zscore_upperbound and not node1 in L:
                    L.append (node1)
                else:
                    print(f"pruned node {node1} with z-score {node1_zscore: 0.4f}")
                    pruned_node_count += 1
                ## node2
                if node2_zscore >= zscore_lowerbound and node2_zscore <= zscore_upperbound and get_rank_of_list (node2, gap_mark) == rank and not node2 in R:
                    R.append (node2)
                elif node2_zscore >= zscore_lowerbound and node2_zscore <= zscore_upperbound and not node2 in L:
                    R.append (node2)
                else:
                    print(f"pruned node {node2} with z-score {node2_zscore: 0.4f}")
                    pruned_node_count += 1
                ## process edges
                if node1_zscore >= zscore_lowerbound and node1_zscore <= zscore_upperbound and node2_zscore >= zscore_lowerbound and node2_zscore <= zscore_upperbound:
                    edge = (node1, node2)
                    #edge = (node2, node1)
                try:
                    if edge and not edge in E:
                        E.append (edge)
                except UnboundLocalError:
                    pass
            ## when upperbound z-score pruning is applied
            elif not zscore_upperbound is None: # z-score pruning applied
                ## node1
                if node1_zscore <= zscore_upperbound and not node1 in L:
                    L.append (node1)
                else:
                    print(f"pruned node {node1} with z-score {node1_zscore: 0.4f}")
                    pruned_node_count += 1
                ## node2
                if node2_zscore <= zscore_upperbound and get_rank_of_list (node2, gap_mark) == rank and not node2 in R:
                    R.append (node2)
                elif node2_zscore <= zscore_upperbound and not node2 in L:
                    R.append (node2)
                else:
                    print(f"pruned node {node2} with z-score {node2_zscore: 0.4f}")
                    pruned_node_count += 1
                ## register instance nodes
                if count_items (node2, gap_mark) == 0 and node2 not in instances:
                    instances.append (node2)
                ## process edges
                if node1_zscore <= zscore_upperbound and node2_zscore <= zscore_upperbound:
                    edge = (node1, node2)
                    #edge = (node2, node1)
                try:
                    if edge and not edge in E:
                        E.append (edge)
                except UnboundLocalError:
                    pass
            ## when lowerbound z-score pruning is applied
            elif not zscore_lowerbound is None: # z-score pruning applied
                ## node1
                if node1_zscore >= zscore_lowerbound and not node1 in L:
                    L.append (node1)
                else:
                    print(f"pruned node {node1} with z-score {node1_zscore: 0.4f}")
                    pruned_node_count += 1
                ## node2
                if node2_zscore >= zscore_lowerbound and get_rank_of_list (node2, gap_mark) == rank and not node2 in R:
                    R.append (node2)
                elif node2_zscore >= zscore_lowerbound and not node2 in L:
                    R.append (node2)
                else:
                    print(f"pruned node {node2} with z-score {node2_zscore: 0.4f}")
                    pruned_node_count += 1
                ## process edges
                if node1_zscore >= zscore_lowerbound and node2_zscore >= zscore_lowerbound:
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
                if not node1 in L:
                    L.append (node1)
                ## node2
                if get_rank_of_list (node2, gap_mark) == rank:
                    if not node2 in R:
                        R.append (node2)
                elif not node2 in L:
                    R.append (node2)
                ## process edges
                if node1 and node2:
                    edge = (node1, node2)
                    #edge = (node2, node1)
                if edge and not edge in E:
                    E.append (edge)
            ## register instance nodes
            if count_items (node2, gap_mark) == 0 and node2 not in instances:
                instances.append (node2)


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
    max_node_count_on_layer = max(node_counts_by_layers)

    ## node color setting
    values_for_color = []
    for node in G:
        try:
            z_value = zscores[node]
            if check:
                print(f"#z_value: {z_value: 0.4f}")
            z_normalized = normalize_score(z_value, use_robust_zscore = use_robust_zscore)
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
                (round(2.5 * len(D), 0),
                round(2 * math.log (max_node_count_on_layer), 0))
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
    used_labels = [ as_label (x, sep = ",") for x in sorted (instances) ]
    label_count = len (used_labels)
    if label_sample_n is not None and label_count > label_sample_n:
        new_labels = used_labels[:label_sample_n - 1]
        new_labels.append("…")
        new_labels.append(used_labels[-1])
        used_labels = new_labels
    print(f"#used_labels: {used_labels}")

    ### set title
    if generalized:
        if use_robust_zscore:
            title_val = f"gPatternLattice (layout: {layout_name}; robust z-scores: {zscore_lowerbound} – {zscore_upperbound}) built from\n{used_labels} ({label_count} in all)"
        else:
            title_val = f"gPatternLattice (layout: {layout_name}; normal z-scores: {zscore_lowerbound} – {zscore_upperbound}) built from\n{used_labels} ({label_count} in all)"
    else:
        if use_robust_zscore:
            title_val = f"PatternLattice (layout: {layout_name}; robust z-scores: {zscore_lowerbound} – {zscore_upperbound}) built from\n{used_labels} ({label_count} in all)"
        else:
            title_val = f"PatternLattice (layout: {layout_name}; normal z-scores: {zscore_lowerbound} – {zscore_upperbound}) built from\n{used_labels} ({label_count} in all)"
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

##
def merge_patterns_and_filter (A, B, check = False):
    return A.merge_patterns (B, check = check)
    #C = A.merge_patterns (B, check = check)
    #return C
    #if not C is None and form_is_None_free (C):
    #    #yield C # fails
    #    return C

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
    coeff     = 0.6745
    ##
    if stdev == 0 or MAD == 0:
        return 0
    else:
        if robust:
            return (coeff * (value - median)) / MAD
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
def normalize_score (x: float, use_robust_zscore: bool = False, min_val: float = -3, max_val: float = 6) -> float:
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
    for i, link_source in enumerate(Link_sources):
        value  = Link_sources[link_source]
        rank   = get_rank_of_list (link_source, gap_mark = gap_mark)
        if use_robust_zscore:
            zscore = calc_zscore (value, averages_by_rank[rank], stdevs_by_rank[rank], medians_by_rank[rank], MADs_by_rank[rank], robust = True)
        else:
            zscore = calc_zscore (value, averages_by_rank[rank], stdevs_by_rank[rank], medians_by_rank[rank], MADs_by_rank[rank], robust = False)
        source_zscores[link_source] = zscore
        print(f"#source {i:3d}: {link_source} has {value} link(s) [{source_zscores[link_source]: .4f} at rank {rank}]")

    ## attach source_zscores to M
    M.source_zscores.update(source_zscores)
    if check:
        print(f"M.source_zscores: {M.source_zscores}")
    ##
    return M

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
        print(f"#target {i:3d}: {link_target} has {value} link(s) [{target_zscores[link_target]: .4f} at rank {rank}]")

    ## attach source_zscores to M
    M.target_zscores.update(target_zscores)
    if check:
        print(f"M.target_zscores: {M.target_zscores}")
    ##
    return M

## Classes
##
class PatternLattice():
    "definition of PatternLattice class"
    ##
    def __init__ (self, pattern, generalized: bool, reflexive: bool = True, check: bool = False):
        "initialization of a PatternLattice"
        if check:
            print(f"pattern.paired: {pattern.paired}")
        ##
        self.origin       = pattern
        self.generalized  = generalized
        if not pattern.gap_mark is None or not pattern.gap_mark == "":
            self.gap_mark = pattern.gap_mark
        else:
            raise "Error: gap_mark is missing"
        self.nodes        = pattern.build_lattice_nodes (generalized = generalized, check = check)
        self.ranked_nodes = self.group_nodes_by_rank (check = check)
        self.links, self.link_sources, self.link_targets = \
            self.gen_links (reflexive = reflexive, check = check)
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
    def print (self):
        out = f"{type(self).__name__} ({self.nodes!r})\n"
        out += f"{type(self).__name__} ({self.source_zscores!r})\n"
        return out

    ##
    def __iter__(self):
        for x in self.nodes:
            yield x

    ##
    def group_nodes_by_rank (self, check: bool = False) -> dict:
        "takes a list of patterns, P, and generates a dictionary of patterns grouped by their ranks"
        import collections
        ##
        gap_mark  = self.gap_mark
        N         = self.nodes
        size      = len(N)
        ## implementation using itertooks.groupby() failed
        rank_finder = lambda p: len([ x for x in p.form if x != gap_mark ])
        ## main
        rank_groups = collections.defaultdict(list) # dictionary
        for pattern in sorted (N, key = rank_finder):
            pattern_rank = pattern.get_rank ()
            if check:
                print(f"#rank: {pattern_rank}")
                print(f"#ranked pattern: {pattern}")
            if pattern_rank <= size:
                rank_groups[pattern_rank].append(pattern)
            if check:
                print(f"#rank_groups: {rank_groups}")
        ##
        return rank_groups

    ##
    def gen_links (self, reflexive: bool, reductive: bool = True, check: bool = False):
        "takes a PatternLattice, extracts ranked_nodes, and generates a list of links among them"
        ##
        if check:
            print(f"#generating links ...")
        ##
        links = [ ]; seen = [ ]
        link_sources, link_targets = defaultdict(int), defaultdict(int)
        #link_sources, link_targets = {}, {}
        G = self.ranked_nodes
        ## main to be multiprocessed
        for rank in sorted (G.keys()):
            if check:
                print(f"#rank: {rank}")
            ##
            L = G[rank]
            if check:
                print(f"#L: {list(L)}")
            ## define R
            try:
                R = G[rank + 1]
                if check:
                    print(f"#R: {list(R)}")
                if reflexive:
                    R = make_simplest_list (L, R)
            except KeyError:
                pass

            ## main: L and R are given
            sub_links = [ ]
            ## define L
            try:
                for l in L: # l is a Pattern
                    l_form, l_content = l.form, l.content
                    if len(l_form) == 0:
                        continue
                    for r in R: # r is a Pattern
                        r_form, r_content = r.form, r.content
                        if check:
                            print(f"#linking r_form: {r_form}; r_content: {r_content}")
                        ## main
                        if len(r_form) == 0:
                            continue
                        elif l_form == r_form:
                            continue
                        elif r.instantiates_or_not (l, check = check):
                            print(f"#is-a: {l_form} <- {r_form}")
                            link = PatternLink((l, r))
                            ##
                            #if not link in links:
                            if not link in sub_links:
                            #if not link in sub_links and not link in sub_links_pre1:
                            #if not link in seen:
                            #if mp_test_for_membership (link, links): # This slows down
                                #links.append (link)
                                sub_links.append (link)
                                link_sources[l_form] += 1
                                link_targets[r_form] += 1
                                ## update seen
                                #seen.append(link)
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
    def update_links (self, reflexive: bool = True, reductive: bool = True, check: bool = False):
        "update links"
        L, L_sources, L_targets = self.gen_links (reflexive = reflexive, reductive = reductive, check = check)
        if check:
            print(f"#L (in update): {L}")
        ##
        self.links = L
        self.link_sources = L_sources
        self.link_targets = L_targets
        return self


    ##
    #@jit(nopython = True)
    def merge_lattices (self, other, gen_links_internally: bool, reflexive: bool, generalized: bool = True, reductive: bool = True, remove_None_containers: bool = False, show_steps: bool = False, use_multiprocess: bool = True, check: bool = False):
        "takes a pair of PatternLattices and returns its merger"
        ##
        print(f"#merging pattern lattices ...")
        if len(other.nodes) == 0:
            return self
        elif len(self.nodes) == 0:
            return other
        ##
        import itertools # This code needs to be externalized under jit
        import os
        ##
        sample_pattern = self.nodes[0]
        gap_mark       = sample_pattern.gap_mark

        ## creates .nodes
        pooled_nodes = self.nodes
        nodes_to_add = other.nodes
        ## remove None-containers
        #if remove_None_containers:
        #    pooled_nodes = [ node for node in pooled_nodes if form_is_None_free (node) ]
        #    nodes_to_add = [ node for node in other.nodes if form_is_None_free (node) ]
        if check:
            print(f"#pooled_nodes [0]: {pooled_nodes}")

        ## adding
        if reductive:
            for node in nodes_to_add:
                ## The following fails to work if Pattern.__eq__ is not redefined
                if not node in pooled_nodes:
                    pooled_nodes.append(node)
        else:
            pooled_nodes += nodes_to_add
        if check:
            print(f"#pooled_nodes [1]: {pooled_nodes}")

        ## reduce source
        if reductive:
            R = [ ]
            ## The following fails to work if Pattern.__eq__ is not redefined
            for node in pooled_nodes:
                if node not in R:
                    R.append(node)
            pooled_nodes = R
            if check:
                print(f"#pooled_nodes [2]: {pooled_nodes}")
        if check:
            print(f"#pooled_nodes [3]: {pooled_nodes}")

        ## multiprocess(ing) version
        if use_multiprocess:
            print(f"#running in multi-processing mode")
            import os
            import multiprocess as mp
            cores = max(os.cpu_count(), 1)
            with mp.Pool(cores) as pool:
                merged_nodes = pool.starmap (merge_patterns_and_filter, itertools.combinations (pooled_nodes, 2))
                ## The following fails unless Pattern.__eq__ is redefined
                #merged_nodes = [ node for node in merged_nodes if not node is None ]
                #booleans = pool.map (lambda x: not x is None, merged_nodes)
                #merged_nodes = [ node for node, boolean in zip (merged_nodes, booleans) if boolean == True ]
            merged_nodes = simplify_list (merged_nodes)
        ## original slower version
        else:
            merged_nodes = [ ]
            for A, B in itertools.combinations (pooled_nodes, 2):
                C = A.merge_patterns (B, check = False)
                ## The following fails unless Pattern.__eq__ is redefined
                #if form_is_None_free (C) and not C in merged_nodes:
                if not C in merged_nodes: # This fails => worsk after supressing None
                    merged_nodes.append(C)
        if check:
            print(f"#merged_nodes: {merged_nodes}")

        # generate merged PatternLattice
        empty_pat           = Pattern([], gap_mark)
        merged              = PatternLattice (empty_pat, generalized = generalized, reflexive = reflexive)
        merged.nodes        = merged_nodes
        merged.ranked_nodes = merged.group_nodes_by_rank (check = check)
        if check:
            print(f"#merged_ranked_nodes: {merged.ranked_nodes}")

        ## conditionally generates links
        if gen_links_internally:
            merged.links, merged.link_sources, merged.link_targets  = \
                merged.gen_links (reflexive = reflexive, check = check)
        else:
            merged.links, merged.link_sources, merged.link_targets = [], [], []
        ## return result
        if len(merged.links) > 0:
            if show_steps:
                print(f"#merger into {len(merged_nodes)} nodes done")
        ## return result
        #yield merged # fails
        return merged

    ##
    def draw_diagrams (self, generalized: bool, layout: str = None, zscores_from_sources: bool = True, auto_fig_sizing: bool = False, zscore_lowerbound: float = None, zscore_upperbound: float = None, use_robust_zscore: bool = False, scale_factor: float = 3, fig_size: tuple = None, label_size: int = None, label_sample_n: int = None, node_size: int = None, font_name: str = None, use_pyGraphviz: bool = False, test: bool = False, check: bool = False) -> None:
        """
        draw a lattice digrams from a given PatternLattice L by extracting L.links
        """
        ##
        #generalized = self.generalized
        links       = self.links
        if check:
            print(f"#links: {links}")
        ##
        ranked_links = make_PatternLinks_ranked (links)
        if check:
            for rank, links in ranked_links.items():
                print(f"#links at rank {rank}:\n{links}")

        ## handle z-scores
        if zscores_from_sources:
            zscores = self.source_zscores
        else:
            zscores = self.target_zscores
        if check:
            i = 0
            for node, v in zscores.items():
                i += 1
                print(f"node {i:4d} {node} has z-score {v:.4f}")

        ## draw PatternLattice
        draw_network (ranked_links.items(), generalized = generalized, layout = layout, fig_size = fig_size, auto_fig_sizing = auto_fig_sizing, node_size = node_size, scale_factor = scale_factor, label_sample_n = label_sample_n, font_name = font_name, zscores = zscores, use_robust_zscore = use_robust_zscore, zscore_lowerbound = zscore_lowerbound, zscore_upperbound = zscore_upperbound, check = check)

### end of file
