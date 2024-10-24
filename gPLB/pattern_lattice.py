## import standard libraries
import collections
import itertools
import numpy as np
import matplotlib
import math

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

##
def merge_lattice_main (nodes, check: bool = False) -> list:
    "takes a pair of pattern lattices and returns their merger"
    merged_nodes = [ ]
    for A, B in itertools.combinations (nodes, 2):
        C = A.merge_patterns (B, check = check)
        ## The following fails to work if Pattern.__eq__ is not redefined
        #if not C in merged_nodes: # This fails.
        if is_None_free (C) and not C in merged_nodes:
            merged_nodes.append(C)
    return merged_nodes

##
def make_ranked_dict (L: list) -> dict:
    "takes a list of lists and returns a dict whose keys are ranks of the lists"
    ranked_dict = {}
    for rank in set([ get_rank_of_list (x) for x in L ]):
        ranked_dict[rank] = [ x for x in L if Pattern(x).get_rank() == rank ]
    ##
    return ranked_dict

##
def get_rank_dists (link_dict: dict, check: bool = False):
    "calculate essential statistics of the rank distribution given"
    ##
    ranked_links = make_ranked_dict (link_dict)
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
def calc_averages_by_rank (link_dict, check: bool = False):
    "calculate averages per rank"
    ##
    ranked_links = make_ranked_dict (link_dict)
    if check:
        print(f"#ranked_links: {ranked_links}")
    ##
    averages_by_rank = {}
    for rank in ranked_links:
        members = ranked_links[rank]
        #print(f"#members: {members}")
        dist = [ link_dict[m] for m in members ]
        #print(f"#dist: {dist}")
        averages_by_rank[rank] = sum(dist)/len(dist)
    ##
    return averages_by_rank

def calc_stdevs_by_rank (link_dict, check: bool = False):
    "calculate stdevs per rank"
    import numpy as np
    ##
    ranked_links = make_ranked_dict (link_dict)
    if check:
        print(f"#ranked_links: {ranked_links}")
    ##
    stdevs_by_rank = {}
    for rank in ranked_links:
        members = ranked_links[rank]
        dist = [ link_dict[m] for m in members ]
        #print(f"dist: {dist}")
        stdevs_by_rank[rank] = np.std(dist)
    ##
    return stdevs_by_rank

##
def calc_medians_by_rank (link_dict, check: bool = False):
    "calculate stdevs per rank"
    import numpy as np
    ##
    ranked_links = make_ranked_dict (link_dict)
    if check:
        print(f"#ranked_links: {ranked_links}")
    ##
    medians_by_rank = {}
    for rank in ranked_links:
        members = ranked_links[rank]
        dist = [ link_dict[m] for m in members ]
        #print(f"dist: {dist}")
        medians_by_rank[rank] = np.median(dist)
    ##
    return medians_by_rank

##
def calc_MADs_by_rank (link_dict, check: bool = False):
    "calculate stdevs per rank"
    import numpy as np
    import scipy.stats as stats
    ##
    ranked_links = make_ranked_dict (link_dict)
    if check:
        print(f"#ranked_links: {ranked_links}")
    ##
    MADs_by_rank = {}
    for rank in ranked_links:
        members = ranked_links[rank]
        dist = [ link_dict[m] for m in members ]
        #print(f"#dist: {dist}")
        MADs_by_rank[rank] = np.median(stats.median_abs_deviation(dist))
    ##
    return MADs_by_rank

##
def calc_zscore (value, average, stdev, median, MAD, robust: bool = True):
    "returns the z-scores of a value against average, stdev, median, and MAD given"
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
def calc_zscore_old (value, average_val, stdev_val):
    "returns z-score given a triple of value, average and stdev"
    if stdev_val == 0:
        return 0
    else:
        return (value - average_val) / stdev_val

##
def normalize_score (x, min_val: float = -4, max_val: float = 7):
    "takes a value in the range of min, max and returns its normalized value"
    import matplotlib.colors as colors
    ##
    normalizer = colors.Normalize(vmin = min_val, vmax = max_val)
    return normalizer(x)

##
def draw_network (D: dict, layout: str, fig_size: tuple = None, auto_fig_sizing: bool = False, label_size: int = None, label_sample_n: int = None, node_size: int = None, zscores: dict = None, zscore_lowerbound = None, scale_factor: float = 3, font_name: str = None, test: bool = False, use_pyGraphviz: bool = False, check: bool = False):
    "draw layered graph under multipartite setting"
    import networkx as nx
    import math
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import seaborn as sns

    ## define graph
    #G = nx.Graph() # does not accept connectionstyle specification
    G = nx.DiGraph()
    instances = [ ] # register instances
    node_dict = { }
    node_counts_by_layers = [ ]
    ##
    for rank, links in sorted(D, reverse = False):
        #print(f"#rank {rank}: {links}")
        L, R, E = [ ], [ ], [ ]
        for link in links:
            if check:
                print(f"#adding link: {link}")
            ## process nodes
            gap_mark      = link.gap_mark
            node1, node2  = link.form_paired
            ## convert lists to tuples to use them as hash keys
            node1 = as_tuple(node1)
            node2 = as_tuple(node2)
            ## add nodes
            if not node1 in L:
                L.append (node1)
            #if not node2 in R:
            if not node2 in R and not node2 in L: # Crucially, "and not node2 in L"
                R.append (node2)
            ## register for instances
            if count_gaps (node2, gap_mark) == 0 and node2 not in instances:
                instances.append (node2)
            ## process edges
            edge = (node1, node2)
            if not edge in E:
                E.append(edge)
        ## populates nodes for G
        G.add_nodes_from (L, rank = rank)
        G.add_nodes_from (R, rank = rank + 1)
        node_counts_by_layers.append (len(R))
        ## populates edges for G
        G.add_edges_from (E)
    ##
    max_node_count_on_layer = max(node_counts_by_layers)

    ## node color setting
    values_for_color = []
    for node in G:
        node_as_tuple = tuple(node)
        try:
            z_value = zscores[node_as_tuple]
            if check:
                print(f"#z_value: {z_value}")
            z_normalized = normalize_score(z_value)
            if check:
                print(f"#z_normalized: {z_normalized}")
            values_for_color.append (z_normalized)
        except KeyError:
            values_for_color.append (0.5) # normalized value falls between 0 and 1.0

    ## filter insignificant nodes
    prune_count = 0
    if not zscore_lowerbound is None:
        print(f"#pruning nodes with z-score less than {zscore_lowerbound}")
        nodes_to_keep = []
        nodes_to_keep_ids = []
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
                print(f"#pruning {node} with z-score {z_value:.6f}")
                prune_count += 1
        print(f"#pruned {prune_count} nodes")

        ## remove nodes
        G.remove_nodes_from ([ x for x in G if not x in nodes_to_keep ])
        ## set node colors
        values_for_color_filtered = [ ]
        for i, value in enumerate(values_for_color):
            if i in nodes_to_keep_ids:
                values_for_color_filtered.append (value)
        ##
        values_for_color = values_for_color_filtered

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
            positions   = nx.multipartite_layout(G, subset_key = "rank")
        ##
        elif layout in [ 'Graphviz', 'graphviz', 'G' ] :
            layout_name = "Graphviz"
            positions   = nx.nx_pydot.graphviz_layout(G) # obsolete?
            #positions = nx.nx_agraph.graphviz_layout(G)
        ##
        elif layout in ['arf', 'ARF' ] :
            layout_name = "Bread-First Search"
            positions   = nx.arf_layout(G, scaling = scale_factor)
        ##
        elif layout in ['bfs', 'BFS' ] :
            layout_name = "Bread-First Search"
            positions   = nx.bfs_layout(G, start = random.choice(G), scale = scale_factor)
        ##
        elif layout in ['Planar', 'planar', 'P'] :
            layout_name = "Planar"
            positions   = nx.planar_layout(G, scale = scale_factor, dim = 2)
        ##
        elif layout in [ 'Fruchterman-Reingold', 'Fruchterman_Reingold', 'fruchterman_reingold', 'FR']:
            layout_name = "Fruchterman-Reingold"
            positions   = nx.fruchterman_reingold_layout (G, scale = scale_factor, dim = 2)
        ##
        elif layout in [ 'Circular', 'circular', 'C' ]:
            layout_name = "Circular"
            positions   = nx.circular_layout (G, scale = scale_factor, dim = 2)
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
        elif layout in [ 'Kamada-Kawai', 'Kamada_Kawai', 'kamda_kawai', 'KK' ]:
            layout_name = "Kamada-Kawai"
            positions   = nx.kamada_kawai_layout (G, scale = scale_factor, dim = 2)
        ##
        else:
            print(f"Unknown layout: Multi-partite (default) is used")
            layout_name = "Multi-partite"
            positions   = nx.multipartite_layout (G, subset_key = "rank")

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
            fig_size_local = (round(2.5 * len(D), 0), round(0.2 * max_node_count_on_layer, 0))
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
                font_size = 8
    else:
        if not label_size is None:
            font_size = label_size
        else:
            font_size = 8
    print(f"#font_size: {font_size}")

    ## set node_size
    if node_size is None:
        node_size = 12
    else:
        try:
            node_size = round(1.2 * node_size/math.log (max_node_count_on_layer), 0)
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

    ## revserses the arrows
    #G = G.reverse(copy = False) # ineffective??

    ## finally draw
    nx.draw_networkx (G, positions,
        #ax = ax1,
        font_family = font_family,
        font_color = 'darkblue', # label font color
        verticalalignment = "bottom", horizontalalignment = "right",
        min_source_margin = 6, min_target_margin = 6,
        font_size = font_size, node_size = node_size,
        node_color = values_for_color, cmap = my_cmap,
        edge_color = 'gray', width = 0.2, arrowsize = 7,
        connectionstyle = connectionstyle,
    )

    ### set title
    ## set labels used in title
    labels = [ x for x in sorted(instances) ]
    if not label_sample_n is None:
        labels = labels[:label_sample_n - 1] + [("...")] + labels[-1]
    plt.title(f"PatternLattice (layout: {layout_name}) built from {labels}")
    ##
    plt.show()

##
class PatternLattice:
    "definition of PatternLattice class"
    ##
    def __init__ (self, pattern, reflexive: bool, track_content: bool = False, generalized: bool = True, check: bool = False):
        "initialization of a PatternLattice"
        if check:
            print(f"pattern.paired: {pattern.paired}")
        ##
        self.origin       = pattern
        self.gap_mark     = pattern.gap_mark
        self.nodes        = pattern.build_lattice_nodes (generalized = generalized, check = check)
        self.ranked_nodes = self.group_by_rank (check = check)
        self.links, self.link_sources, self.link_targets = \
            self.gen_links (reflexive = reflexive, track_content = track_content, check = check)
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
    def group_by_rank (self, check: bool = False):
        " takes a list of patterns, P, and generates a dictionary of patterns grouped by their ranks"
        ##
        gap_mark = self.gap_mark
        N        = self.nodes
        size = len(N)
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
    def merge_lattices (self, other, gen_links: bool, reflexive: bool, generalized: bool = True, reductive: bool = True, remove_None_containers: bool = False, show_steps: bool = False, check: bool = False):
        ## creates .nodes
        pooled_nodes = self.nodes
        nodes_to_add = other.nodes
        ## remove None-containers
        if remove_None_containers:
            pooled_nodes = [ node for node in pooled_nodes if is_None_free (node) ]
            nodes_to_add = [ node for node in other.nodes if is_None_free (node) ]
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
        ##
        if check:
            print(f"#pooled_nodes [3]: {pooled_nodes}")
        ## Is the following multiprocessible?
        merged_nodes = [ ]
        for A, B in itertools.combinations (pooled_nodes, 2):
            C = A.merge_patterns (B, check = False)
            ## The following fails to work if Pattern.__eq__ is not redefined
            #if not C in merged_nodes: # This fails.
            if is_None_free (C) and not C in merged_nodes:
                merged_nodes.append(C)
        ##
        if check:
            print(f"#merged_nodes: {merged_nodes}")
        # generate merged PatternLattice
        merged   = PatternLattice (Pattern([]), generalized = generalized, reflexive = reflexive)
        merged.nodes        = merged_nodes
        merged.ranked_nodes = merged.group_by_rank (check = check)
        ## conditionally generates links
        if gen_links:
            merged.links, merged.link_sources, merged.link_targets  = \
                merged.gen_links (reflexive = reflexive, check = check)
        else:
            merged.links, merged.link_sources, merged.link_targets = [], [], []
        ## return result
        if show_steps:
            print(f"#Merger into {len(merged_nodes)} nodes done")
        return merged

    ##
    def gen_links (self, reflexive: bool, track_content: bool = False, reductive: bool = True, check: bool = False):
        "takes a PatternLattice, extracts ranked_nodes, and generates a list of links among them"
        ##
        G = self.ranked_nodes
        links = [ ]
        link_sources, link_targets = {}, {}
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
                if check:
                    print(f"#R: {list(R)}")
                ## main
                if reflexive:
                    R = make_simplest_list(L, R)
                ##
                # put multiprocessing process here
                for l in L:
                    ## l is a Pattern
                    l_form, l_content = l.form, l.content
                    ## main
                    if len(l_form) == 0:
                        pass
                    for r in R:
                        ## r is a Pattern
                        r_form, r_content = r.form, r.content
                        if check:
                            print(f"#linking r_form: {r_form}; r_content: {r_content}")
                        ## main
                        if len(r_form) == 0:
                            continue
                        ##
                        if l_form == r_form:
                            continue
                        elif r.instantiates_or_not (l, check = check):
                            print(f"#instantiated {l.form} by {r.form}")
                            link = PatternLink([l, r])
                            ##
                            if not link in links:
                                ## register for links
                                links.append (link)
                                ## register for link_sources, link_targets
                                l_sig = as_tuple(l.form)
                                r_sig = as_tuple(r.form)
                                try:
                                    link_sources[l_sig] += 1
                                    link_targets[r_sig] += 1
                                except KeyError:
                                    link_sources[l_sig] = 1
                                    link_targets[r_sig] = 1
            ##
            except KeyError:
                pass
        ##
        return links, link_sources, link_targets

    ##
    def gen_ranked_links (self, reflexive: bool, reductive: bool = True, check: bool = False):
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
                if check:
                    print(f"#linking l_form: {l_form}; l_content: {l_content}")
                for r in R:
                    r_form, r_content = r.form, r.content
                    if check:
                        print(f"#linking l_form: {l_form}; l_content: {l_content}")
                        print(f"#linking r_form: {r_form}; r_content: {r_content}")
                    ## main
                    if l_form == r_form:
                        pass
                    elif r.instantiates_or_not (l):
                        link = PatternLink([l, r])
                    else:
                        pass
                    ##
                    if link:
                        if check:
                            print(f"#instatiation: True")
                        rank = link.get_rank()
                        try:
                            if link and not link in ranked_links[rank]:
                                ranked_links[rank].append(link)
                        except KeyError:
                            ranked_links[rank] = [link]
        ##
        return ranked_links

    ##
    def update_links (self, reflexive: bool, check: bool = False):
        "update links"
        L = self.gen_links (reflexive = reflexive, check = check)
        if check:
            print(f"#L (in update): {L}")
        self.links = L
        return self

    ##
    def draw_diagrams (self, layout: str = None, get_zscores_from_targets: bool = False, auto_fig_sizing: bool = False, zscore_lowerbound: float = None, scale_factor: float = 3, fig_size: tuple = None, label_size: int = None, label_sample_n: int = None, node_size: int = None, font_name: str = None, use_pyGraphviz: bool = False, test: bool = False, check: bool = False) -> None:
        """
        draw a lattice digrams from a given PatternLattice L by extracting L.links
        """
        ##
        links = self.links
        if check:
            print(f"#links: {links}")
        ##
        ranked_links = make_PatternLinks_ranked (links)
        if check:
            for rank, links in ranked_links.items():
                print(f"#links at rank {rank}:\n{links}")

        ## handle z-scores
        if get_zscores_from_targets:
            zscores = self.target_zscores
        else:
            zscores = self.source_zscores
        if check:
            i = 0
            for k, v in zscores.items():
                i += 1
                print(f"node {i} {k} has z-score {v:.5f}")

        ## draw PatternLattice
        draw_network (ranked_links.items(), layout = layout, fig_size = fig_size, auto_fig_sizing = auto_fig_sizing, node_size = node_size, zscores = zscores, zscore_lowerbound = zscore_lowerbound, scale_factor = scale_factor, font_name = font_name, check = check)

### end of file
