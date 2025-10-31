#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#
"""
gPLB

A Python implementation of generalized Pattern Lattice Builder (gPLB)

developed by Kow Kuroda

"Generalized" means that a pattern lattice build from [a, b, c] includes either [_, a, b, c], [a, b, c, _] and [_, a, b, c, _] (Level 1 generalization) or ['_', 'a', '_', 'b', 'c'], ['a', '_', 'b', 'c', '_'], ['_', 'a', '_', 'b', 'c', '_'], ['_', 'a', 'b', '_', 'c'], ['a', 'b', '_', 'c', '_'], ['_', 'a', 'b', '_', 'c', '_'], ['_', 'a', '_', 'b', '_', 'c'], ['a', '_', 'b', '_', 'c', '_'], ['_', 'a', '_', 'b', '_', 'c', '_'] (Level 2 generalization). Level 1 generalization is concerned with gaps at edges only, whereas Level 2 generalization with all possible insertion points. This makes pyPLB different from RubyPLB (rubyplb) developed by Yoichoro Hasebe and Kow Kuroda, available at <https://github.com/yohasebe/rubyplb>.

created on 2024/09/24

modification history
2024/10/11 fixed a bug in instantiates(), added make_R_reflexive
2024/10/12, 13 added z-score calculation
2024/10/15 completed z-score based coloring of nodes
2024/10/16 fixed a bug in instantiation, remove make_R_reflexive
2024/10/17 fixed a bug in failing connected graph: check for content compatibility offensive; implemented curved edges
2024/10/18 improved font size, node size manipulation; added Japanese font capability
2024/10/20 added package capability
2024/10/21 improved instantiation implementation
2024/10/23 implemented robust z-score, experimented hash-based comparison, fixed wrong layering of some nodes
2024/10/24 fixed a bug in pattern sizing that result improper alignment of some patterns
2024/10/31 fixed a bug in multiparite layout; implemented content tracking on variables in a pattern
2024/11/01 implemented upperbound of z-score pruning of lattice nodes
2024/11/02 improved implementation of Pattern: .form and .content are tuples rather than lists. This change is intended to memory conservation.
2024/11/01 fixed bugs introduced at merger_lattices.
2024/12/02 fixed a bug to produce wrong laying of nodes
2024/12/03 finished implementation of multiprocess version; implemented (un)capitalization of elements; implemented removal of punctuation marks; implemented subsegmentation of hyphenated tokens
2025/01/06 improved the handling of input so that the script now accepts i) # and % for comment escapes, and ii) regex to field separator.
2025/01/07 better auto figure sizing is implemented; mark_instances option is implemented;
2025/09/04 added recursion limit increase;
2025/09/16 modified auto_figsizing;
2025/09/17 changed default value for -G;
2025/09/18 fixed a bug in rank calculation;
2025/09/30 implemented generalization level 2;
2025/10/03 refactored graph drawing algorithm but Multiparite still fails under obscure conditions;
2025/10/04 added MPG_key option that allows changing 'subset_key' in Multi-partite graph;
2025/10/09 fixed a bug to calc_zscore() to get robust z-scores;
2025/10/14 retyped Pattern.form and Pattern.content as tuples, making as_tuple() dispensable; implemented gap_size-based z-score calculation;
2025/10/24 added alternative use of input_field_seps ",;": if sep2_is_suppressive is True, segmentation by "," is suppressed, thereby implementing segmentation on a larger scale;
2025/10/25 implemented truncation in input: "a(b),c" is treated as "a,c" while node "a(b),c" appears at node;
2025/10/29 moved z-score filtering from gen_G to draw_graph (using subgraph() of NetworkX); implemented draw_lattice and made it = False default behavior [changeable by -D option];
2025/10/30 added handling input name to output figure name;
2025/10/31 changed default behavior: gPLB now saves GML by default without drawing; use -D to enable drawing

"""

#
## modules to use
import re
import functools
import pprint as pp
import random

## memory monitoring
import psutil
import os
def print_memory_usage (label=""):
    """Print current memory usage"""
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    print(f"\n## Memory usage {label}: {mem_mb:.1f} MB")

## networkx
import networkx as nx

def pattern_lattice_to_gml (lattice, output_file: str, 
                           include_zscores: bool = True,
                           check: bool = False):
    """Save PatternLattice as GML format"""
    
    G = nx.DiGraph()
    
    # Graph metadata
    G.graph['generality'] = lattice.generality
    G.graph['p_metric'] = lattice.p_metric
    G.graph['gap_mark'] = lattice.gap_mark
    G.graph['tracer'] = lattice.tracer if hasattr(lattice, 'tracer') else '~'
    G.graph['n_nodes'] = len(lattice.nodes)
    G.graph['n_links'] = len(lattice.links)
    
    # Prepare z-scores
    source_zscores = {}
    target_zscores = {}
    source_robust_zscores = {}
    target_robust_zscores = {}
    
    if include_zscores:
        if hasattr(lattice, 'source_zscores'):
            source_zscores = {str(k): float(v) for k, v in lattice.source_zscores.items()}
        if hasattr(lattice, 'target_zscores'):
            target_zscores = {str(k): float(v) for k, v in lattice.target_zscores.items()}
        if hasattr(lattice, 'source_robust_zscores'):
            source_robust_zscores = {str(k): float(v) for k, v in lattice.source_robust_zscores.items()}
        if hasattr(lattice, 'target_robust_zscores'):
            target_robust_zscores = {str(k): float(v) for k, v in lattice.target_robust_zscores.items()}
    
    # Add nodes
    for pattern in lattice.nodes:
        node_id = str(tuple(pattern.form))
        
        node_attrs = {
            'label': ' '.join(pattern.form),
            'form': ','.join(pattern.form),
            'rank': pattern.get_rank(),
            'gap_size': pattern.get_gap_size(),
            'size': len(pattern.form)
        }
        
        # Add z-scores
        if include_zscores:
            if node_id in source_zscores:
                node_attrs['source_zscore'] = source_zscores[node_id]
            if node_id in target_zscores:
                node_attrs['target_zscore'] = target_zscores[node_id]
            if node_id in source_robust_zscores:
                node_attrs['source_robust_zscore'] = source_robust_zscores[node_id]
            if node_id in target_robust_zscores:
                node_attrs['target_robust_zscore'] = target_robust_zscores[node_id]
        
        G.add_node(node_id, **node_attrs)
    
    # Add edges
    for link in lattice.links:
        source_id = str(tuple(link.left.form))
        target_id = str(tuple(link.right.form))
        
        edge_attrs = {
            'link_type': link.link_type if link.link_type else 'instantiates',
            'link_rank': link.get_link_rank(),
            'link_gap_size': link.get_link_gap_size()
        }
        
        G.add_edge(source_id, target_id, **edge_attrs)
    
    # Save
    nx.write_gml(G, output_file)
    
    if check:
        print(f"✓ Saved lattice to GML: {output_file}")
        print(f"  Nodes: {G.number_of_nodes()}")
        print(f"  Edges: {G.number_of_edges()}")
    
    return output_file


## argument parsing
import argparse
def parse_tuple_for_arg (s: str, sep: str = ',') -> tuple:
    """Converts a string of comma-separated values into a tuple of integers."""
    try:
        return tuple(int(x.strip()) for x in s.split(sep))
    except ValueError:
        raise argparse.ArgumentTypeError("Tuple values must be integers separated by commas.")

##
parser  = argparse.ArgumentParser(description = "")
parser.add_argument('file', type=open, default=None)
parser.add_argument('-v', '--verbose', action='store_true', default=False)
parser.add_argument('-w', '--detailed', action='store_true', default=False)
parser.add_argument('-M', '--use_mp', action='store_false', default=True)
parser.add_argument('-N', '--print_forms_only', action='store_true', default=False)
parser.add_argument('-O', '--print_lattice', action='store_true', default=False)
parser.add_argument('-D', '--save_lattice', action='store_false', default=True,
                    help='Draw lattice after building (default: only save GML)')
parser.add_argument('--output_gml', '-o', type=str, default=None,
                    help='GML output filename (default: auto-generate from input)')
parser.add_argument('--no_gml', action='store_true', default=False,
                    help='Disable GML output (for compatibility)')
parser.add_argument('-F', '--fig_size', type=parse_tuple_for_arg, default=None)
parser.add_argument('-d', '--fig_dpi', type=int, default=620)
parser.add_argument('-L', '--layout', type= str, default= 'Multi_partite')
parser.add_argument('-I', '--draw_individual_lattices', action='store_true', default=False)
parser.add_argument('-S', '--build_lattice_stepwise', action='store_true', default=False)
parser.add_argument('-J', '--use_multibyte_chars', action='store_true', default=False)
parser.add_argument('-c', '--input_comment_escapes', type=list, default=['#', '%'])
parser.add_argument('-s', '--input_field_seps', type=str, default=',;')
parser.add_argument('-P', '--sep2_is_suppressive', action='store_true', default=False)
parser.add_argument('-C', '--uncapitalize', action='store_true', default=False)
parser.add_argument('-H', '--split_hyphenation', action='store_false', default=True)
parser.add_argument('-g', '--gap_mark', type=str, default='_')
parser.add_argument('-t', '--tracer', type=str, default='~')
parser.add_argument('-Q', '--accept_truncation', action='store_false', default=True)
parser.add_argument('-X', '--add_displaced_versions', action='store_true', default=False)
parser.add_argument('-n', '--sample_n', type=int, default=None)
parser.add_argument('-m', '--max_size', type=int, default=None)
parser.add_argument('-G', '--generality', type=int, default=0)
parser.add_argument('--max_patterns', type=int, default=None, help='Maximum patterns per segment')
parser.add_argument('--batch_size', type=int, default=5000, help='Batch size for link generation')
parser.add_argument('-R', '--unreflexive', action='store_false', default=True)
parser.add_argument('-p', '--productivity_metric', type=str, default='rank')
parser.add_argument('-l', '--zscore_lowerbound', type=float, default=None)
parser.add_argument('-u', '--zscore_upperbound', type=float, default=None)
parser.add_argument('-Z', '--use_robust_zscore', action='store_false', default=True)
parser.add_argument('-k', '--MPG_key', type=str, default='gap_size')
parser.add_argument('-T', '--zscores_from_targets', action='store_true', default=False)
parser.add_argument('-j', '--scaling_factor', type=float, default=5)
parser.add_argument('-i', '--mark_instances', action='store_true', default=False)
parser.add_argument('-U', '--print_link_targets', action='store_true', default=False)
parser.add_argument('-Y', '--phrasal', action='store_true', default=False)
parser.add_argument('--recursion_limit_factor', type=float, default=1.0)
parser.add_argument('--sample_id', type=int, default=1)

##
args = parser.parse_args()
##
file                   = args.file   # process a file when it exists
verbose                = args.verbose
detailed               = args.detailed
recursion_limit_factor = args.recursion_limit_factor
use_mp                 = args.use_mp # controls use of multiprocess
print_lattice          = args.print_lattice
save_lattice           = args.save_lattice
output_gml_file        = args.output_gml
no_gml_output          = args.no_gml
input_comment_escapes  = args.input_comment_escapes
input_field_seps       = args.input_field_seps
sep2_is_suppressive    = args.sep2_is_suppressive # controls the behavior of second sep
accept_truncation      = args.accept_truncation
uncapitalize           = args.uncapitalize
split_hyphenation      = args.split_hyphenation
gap_mark               = args.gap_mark
tracer                 = args.tracer
max_size               = args.max_size
sample_n               = args.sample_n
print_forms_only       = args.print_forms_only
reflexive              = args.unreflexive
generality             = args.generality
max_patterns           = args.max_patterns
batch_size             = args.batch_size
p_metric               = args.productivity_metric
add_displaced_versions = args.add_displaced_versions
build_lattice_stepwise = args.build_lattice_stepwise
print_link_targets     = args.print_link_targets
fig_size               = args.fig_size
layout                 = args.layout
MPG_key                = args.MPG_key
fig_dpi                = args.fig_dpi
zscore_lowerbound      = args.zscore_lowerbound
zscore_upperbound      = args.zscore_upperbound
use_robust_zscore      = args.use_robust_zscore
zscores_from_targets   = args.zscores_from_targets
mark_instances         = args.mark_instances
draw_individually      = args.draw_individual_lattices
use_multibyte_chars    = args.use_multibyte_chars
scale_factor           = args.scaling_factor
phrasal                = args.phrasal
sample_id              = args.sample_id


## script parameters
draw_inspection      = False
draw_inline          = False # intended to be used in Jupyter Notebook

## implications
## increase recursion limit
if recursion_limit_factor != 1.0:
    import sys
    sys.setrecursionlimit(round(recursion_limit_factor * 1000))

## verbosity
if verbose:
    check = True
else:
    check = False

## fig_size
auto_figsizing = False
if fig_size is None:
    auto_figsizing = True

## make_links_safely
if check:
    make_links_safely = True # False previously
else:
    make_links_safely = False

## show paramters
print(f"## Parameters")
print(f"# use_multiprocess: {use_mp}")
print(f"# detailed: {detailed}")
print(f"# verbose: {verbose}")
print(f"# save_lattice: {save_lattice}")
print(f"# draw_inline: {draw_inline}")
print(f"# auto_figsizing: {auto_figsizing}")
print(f"# fig_size: {fig_size}")
print(f"# fig_dpi: {fig_dpi}")
print(f"# draw_individually: {draw_individually}")
print(f"# mark_instances: {mark_instances}")
print(f"# input_comment_escapes: {input_comment_escapes}")
print(f"# input_field_seps: {input_field_seps}")
print(f"# sep2_is_suppressive: {sep2_is_suppressive}")
print(f"# accept_truncation: {accept_truncation}")
print(f"# uncapitalize: {uncapitalize}")
print(f"# split_hyphenation: {split_hyphenation}")
print(f"# gap_mark: {gap_mark}")
print(f"# instantiation is reflexive: {reflexive}")
print(f"# building lattice with generality: {generality}")
print(f"# p_metric [productivity metric]: {p_metric}")
print(f"# use_robust_zscore: {use_robust_zscore}")
print(f"# zscores_from_targets: {zscores_from_targets}")
print(f"# zscore_lowerbound: {zscore_lowerbound}")
print(f"# zscore_upperbound: {zscore_upperbound}")
print(f"# make_links_safely: {make_links_safely}")

## import modules
try:
    ## Try relative imports first
    from .utils import *
    from .pattern import *
    from .pattern_link import *
    from .pattern_lattice import *
except ImportError:
    # Fall back to absolute imports (when run as script)
    import sys
    import os
    # Add current directory to path if needed
    if __name__ == '__main__':
        sys.path.insert(0, os.path.dirname(__file__))
    from utils import *
    from pattern import *
    from pattern_link import *
    from pattern_lattice import *


### Functions

##
def parse_input (file, comment_escapes: list, field_seps: str, split_hyphenation: bool = split_hyphenation, uncapitalize: bool = uncapitalize, check: bool = False) -> list:
    """
    reads a file, splits it into segments using a given separator, removes comments, and forward the result to main
    """

    ## reading data
    with file as f:
        lines =  [ line.strip() for line in f.readlines() if len(line) > 0 ]
    if check:
        print(f"# input: {lines}")

    ## remove inline comments
    filtered_lines = [ strip_comment(line, comment_escapes) for line in lines ]
    if check:
        print(f"# filtered_lines: {filtered_lines}")

    ## generate segmentations
    segmented_lines = [ segment_with_levels (line, seps = field_seps, sep2_is_suppressive = sep2_is_suppressive, split_hyphenation = split_hyphenation, uncapitalize = uncapitalize, check = check)
                for line in filtered_lines if len(line) > 0 ]

    ##
    return segmented_lines

##
def setup_font (
                system_font_dir: str = "/System/Library/Fonts/",
                user_font_dir: str = "/Library/Fonts/",
                user_font_dir2: str = "/usr/local/texlive/2013/texmf-dist/fonts/truetype/public/ipaex/",
                check: bool = False):
    """set font for Japanese character display"""
    import matplotlib
    if use_multibyte_chars:
        from matplotlib import font_manager as Font_manager
        ## select font
        multibyte_font_names = [    "IPAexGothic",  # 0 Multi-platform font
                                    "Hiragino sans" # 1 Mac only
                                ]
        multibyte_font_name  = multibyte_font_names[0]

        # use the version installed via TeXLive
        if multibyte_font_name == "IPAexGothic":
            try:
                Font_manager.fontManager.addfont(f"{user_font_dir}ipaexg.ttf")
            except FileNotFoundError:
                Font_manager.fontManager.addfont(f"{user_font_dir2}ipaexg.ttf")
        elif multibyte_font_name == "Hiragino sans":
            Font_manager.fontManager.addfont(f"{system_font_dir}ヒラギノ角ゴシック W0.ttc")
        ## check result
        matplotlib.rc('font', family = multibyte_font_name)
    else:
        multibyte_font_name = None
        matplotlib.rcParams['font.family'] = "Sans-serif"
    
    ## check font settings
    print(f"## multibyte_font_name: {multibyte_font_name}")
    print(f"## matplotlib.rcParams['font.family']: {matplotlib.rcParams['font.family']}")
    
    ## return
    return multibyte_font_name

## process
S0 = []
input_file_name_stem = None
if not file is None:
    ## define input and output file names
    from pathlib import Path
    input_file_name = Path(file.name)
    input_file_name_stem = Path(file.name).stem
    ##
    if output_gml_file is None and not no_gml_output:
        output_gml_file = f"g{generality}PL-{input_file_name_stem}.gml"

    ## parse source
    input_parses = parse_input (file, comment_escapes = input_comment_escapes, field_seps = input_field_seps, split_hyphenation = split_hyphenation, uncapitalize = uncapitalize, check = False)
    S0.extend (input_parses)
else:
    if phrasal: # phrasal source
        Text1 = [ 'a big boy', 'the big boy', 'a big girl', 'the big girl',
            'a funny boy', 'the funny boy', 'the funny boys', 'funny boys',
            'a small boy', 'a small girl', 'the small boy', 'the small girl',
            'big boys', 'big girls', 'small boys', 'small girls', 'the funny girl',
            'the big boys', 'the small boys', 'the boys', 'the girls' ]
        Phrases1 = [ t.split() for t in Text1 ]
        S0 = Phrases1
    else: # lexical sources
        Words3 = [ "bye", "pie", "lie", "pye", "pip", "pig", "lig", "bug", "hug", "rug", "say", "lye", "dig", "fog", "pin", "sin", "day", "dan", "dye", "may", "way", "fig", "dog" ]
        Words4 = [ "gene", "dean", "fine", "sine", "wine", "wing", "wide", "pine", "line", "mine", "pane", "wane", "wade", "lamb", "womb", "bomb", "find", "wind", "folk", "dogs", "fogs", "bombs", "winds", "finds", "wines", "wings"  ]
        Words5 = [ "power", "pride", "poker", "slide", "tried", "kinky", "image", "ships", "deals", "wings", "folks", "quick" ]
        Words6 = [ "system", "people", "winter", "wider", "ginger", "singer", "winder", "finder", "walker", "talker", "widens" ]
        Words7 = [ "thunder", "grinder", "reminder", "stapler", "quicksand" ]
        Words0 = random.sample(Words3, 5) + random.sample(Words4, 4) + random.sample(Words5, 3) + random.sample(Words6, 4) + random.sample(Words7, 2)
        ## selection
        if   sample_id == 0:
            Words = Words0
        elif sample_id == 1:
            Words = Words3
        elif sample_id == 2:
            Words = Words4
        elif sample_id == 3:
            Words = Words5
        elif sample_id == 4:
            Words = Words6
        elif sample_id == 5:
            Words = Words7
        else:
            raise f"# sample_id {sample_id} is not defined"
        sample_S = [ [ seg for seg in re.split(r"", t) if len(seg) > 0 ] for t in Words ]
        S0.append (sample_S)

## memory monitoring point 1
print_memory_usage ("after reading input")

## filter
if not max_size is None:
    S0 = [ x for x in S0 if len(x) <= max_size and len(x) > 0 ]

## take a sample
if sample_n is not None and len(S0) > sample_n:
    try:
        S = random.sample (S0, sample_n)
    except ValueError:
        S = S0
else:
    S = S0
if verbose:
    print(f"# S: {S}")

## select source
print(f"## Source lists:")
for i, s in enumerate(S):
    print (f"# source {i}: {s}")

## generating patterns
Patterns = [ ]
for s in S:
    ## expand input under generalization
    T = [s]
    ## level 3 generalization
    if generality == 3:
        T.extend (insert_gaps (s, gap_mark = gap_mark))
        if add_displaced_versions:
            T.extend (create_displaced_versions (s, tracer = tracer, gap_mark = gap_mark))
        T.extend (add_gaps_around (s, gap_mark = gap_mark))
    ## level 2 generalization
    elif generality == 2:
        T.extend (insert_gaps (s, gap_mark = gap_mark))
        if add_displaced_versions:
            T.extend (create_displaced_versions (s, tracer = tracer, gap_mark = gap_mark))
    ## level 1 generalization
    elif generality == 1:
        T.extend (add_gaps_around (s, gap_mark = gap_mark))
        if add_displaced_versions:
            T.extend (create_displaced_versions (s, tracer = tracer, gap_mark = gap_mark))
    ## level 0 generalization
    else:
        if add_displaced_versions:
            T.extend (create_displaced_versions (s, tracer = tracer, gap_mark = gap_mark))
    ##
    for s in T:
        print(f"# processing: {s}")
        try:
            p = Pattern(s, gap_mark = gap_mark, accept_truncation = accept_truncation)
        except TypeError:
            p = Pattern(s, gap_mark = gap_mark, tracer = tracer, accept_truncation = accept_truncation)
        if detailed:
            print(f"# p: {p}")
        Patterns.append(p)
##
Patterns = sorted (Patterns, key = lambda x: len(x), reverse = False)
##
for i, pat in enumerate(Patterns):
    if verbose:
        print(f"# gapped patterns from pattern {i}: {pat}")
    for i, g_pat in enumerate(pat.create_gapped_versions (check = False)):
        if verbose:
            print(f"# gapped {i+1}: {g_pat}")
#exit()

## memory monitoring point 2
print_memory_usage ("after generating patterns")

##
print(f"## Generating g{generality}PLs ...")
L = [ ]
for i, p in enumerate(Patterns):
    print(f"# generating g{generality}PL {i+1} from {p}")
    ## main
    patlat = PatternLattice (p, generality = generality, reflexive = reflexive, make_links_safely = make_links_safely, check = False)
    if detailed:
        pp.pprint(patlat)
    ##
    if verbose:
        print(f"# patlat.origin: {patlat.origin}")
        if detailed:
            pp.pprint (patlat.origin)
    ##
    if verbose:
        print(f"# patlat.nodes; count: {len(patlat.nodes)}")
        if detailed:
            pp.pprint (patlat.nodes)
    ##
    if verbose:
        print(f"# patlat.ranked_nodes; count: {len(patlat.ranked_nodes)}")
        if detailed:
            pp.pprint (patlat.ranked_nodes)
    ##
    if verbose:
        print(f"# patlat.links; count: {len(patlat.links)}")
        if detailed:
            pp.pprint (patlat.links)
    ##
    L.append (patlat)
#exit()

## memory monitoring point 3
print_memory_usage ("after building lattices")

##
if detailed:
    for i, patlat in enumerate(L):
        for j, pattern in enumerate(patlat):
            print(f"# p{i}.{j}: {pattern}")
#exit()

## print forms and then quit without drawing lattices
if print_forms_only:
    joint = input_field_seps
    for i, patlat in enumerate(L):
        for j, pat in enumerate(patlat):
            print(f"# p{i:02d}.form{j:03d}: {joint.join(pat.get_form())}")
    ##
    exit()
    
### Generating Pattern Lattices

## draw lattices and then quit without drawing the merged lattice
if draw_individually:
    print(f"## Processing g{generality}PLs individually")
    for i, patlat in enumerate(L):
        print(f"# g{generality}PL {i+1}")
        ## print
        if print_lattice:
            print(f"# Printing a{generality}PL:\n")
            print(patlat.print())
        
        ## draw
        if not save_lattice:
            print(f"# Drawing a diagram from g{generality}PL {i+1}")
            multibyte_font_name = setup_font ()
            patlat.draw_lattice (layout = layout, MPG_key = MPG_key, save_lattice = save_lattice, draw_inline = draw_inline, auto_figsizing = auto_figsizing, fig_size = fig_size, fig_dpi = fig_dpi, generality = generality, p_metric = p_metric, make_links_safely = make_links_safely, zscores_from_targets = zscores_from_targets, mark_instances = mark_instances, scale_factor = scale_factor, font_name = multibyte_font_name, check = draw_inspection)
        else:
            print(f"## Skipped drawing (use -D/--draw-lattice to enable visualization)")
    ##
    exit()

##
print(f"## Merging {len(L)} g{generality}PLs ...")
simplified     = False
label_sample_n = 10
if simplified:
    #print(f"#binary merger")
    La, Lb = L[0], L[1]
    if verbose:
        print(f"# La: {La}")
        print(f"# Lb: {Lb}")
    M = La.merge_with (Lb, use_mp = use_mp, show_steps = True, check = False)

## Individual draw
elif build_lattice_stepwise:
    gen_links_internally = True
    print(f"## Mergig g{generality}PLs ...")
    for i, patlat in enumerate (L):
        print(f"# Processing g{generality}PL {i+1}")
        if i == 0:
            M = patlat
        else: ## merger
            M = M.merge_with (patlat, gen_links_internally = gen_links_internally, use_mp = use_mp, generality = generality, reflexive = reflexive, reductive = True, show_steps = True, check = False)
            ## delete the original
            patplat = None

        ## check nodes in M
        print(f"# merged g{generality}PL with {len(M.nodes)} nodes")
        for i, p in enumerate(M.nodes):
            print(f"# node {i:3d}: {p.separate_print()}")

        ## genenrate links in delay
        if len(M.links) == 0 and not gen_links_internally:
            ## Don't do: M = M.update(...)
            M.update_links (reflexive = reflexive, check = False) ## Crucially
        print(f"# generated {len(M.links)} links")

        ## checking links in M
        print(f"## Links")
        for i, link in enumerate(M.links):
            link.pprint (indicator = i, paired = True, link_type = "instantiates", check = False)

        ## generate z-scores from link targets
        gen_zscores_from_targets_by (p_metric, M, gap_mark = gap_mark, use_robust_zscore = use_robust_zscore, check = False)
        if print_link_targets:
            for node, zscore in M.source_zscores.items():
                print(f"# node {node} has z-score {zscore: .3f}")

        ## generate z-scores from link sources
        gen_zscores_from_sources_by (p_metric, M, gap_mark = gap_mark, use_robust_zscore = use_robust_zscore, check = False)
        for node, zscore in M.source_zscores.items():
            print(f"# node {node} has z-score {zscore: .3f}")
        
        ##
        print(f"## Output")
        ## print lattice
        if print_lattice:
            print(f"## Merged PL:")
            print(M.print())
        
        ## draw lattice
        if not save_lattice:
            multibyte_font_name = setup_font ()
            M.draw_lattice (layout, MPG_key, save_lattice = save_lattice, draw_inline = draw_inline, input_name = input_file_name_stem, auto_figsizing = auto_figsizing, fig_size = fig_size, fig_dpi = fig_dpi, generality = generality, label_sample_n = label_sample_n, p_metric = p_metric, make_links_safely = make_links_safely, use_robust_zscore = use_robust_zscore, zscore_lb = zscore_lowerbound, zscore_ub = zscore_upperbound, mark_instances = mark_instances, font_name = multibyte_font_name, zscores_from_targets = zscores_from_targets, scale_factor = scale_factor, check = draw_inspection)
        else:
            print(f"# Skipped drawing (use -D/--draw-lattice to enable visualization)")

## Draw after integration
else:
    gen_links_internally = False
    M = functools.reduce (lambda La, Lb: La.merge_with (Lb, gen_links_internally = gen_links_internally, use_mp = use_mp, generality = generality, reflexive = reflexive, reductive = True, check = False), L)

    # The following process was isolated for memory conservation
    if len(M.links) == 0 and not gen_links_internally:
        print(f"## Generating links independently")
        ## N.B. 1) Don't do: M = M.update(...); 2) update_links() is rank-based
        M.update_links (reflexive = reflexive, use_mp = use_mp, check = False)

    ##
    print(f"## Results")

    ## check nodes in M
    print(f"# Merger has {len(M.nodes)} nodes")
    for i, p in enumerate(M.nodes):
        print(f"# node {i}: {p}")

    ## check links in M
    print(f"# Merger has {len(M.links)} links")
    for i, link in enumerate(M.links):
        link.pprint (indicator = i, paired = True, link_type = "instantiates", check = False)

    ## get z-scores from link targets
    print(f"## Calculating z-scores ...")
    gen_zscores_from_targets_by (p_metric, M, gap_mark = gap_mark, tracer = tracer, use_robust_zscore = use_robust_zscore, check = False)
    if print_link_targets:
        for node, zscore in M.target_zscores.items():
            print(f"# node {node} has z-score {zscore: .3f} [n: {M.link_targets[node]:2d}]")

    ## get z-scores from link sources
    gen_zscores_from_sources_by (p_metric, M, gap_mark = gap_mark, tracer = tracer, use_robust_zscore = use_robust_zscore, check = False)
    for node, zscore in M.source_zscores.items():
        print(f"# node {node} has z-score {zscore: .3f} (n: {M.link_sources[node]:2d})")

    ##
    print(f"## Output")
    
    ## print lattice optionally
    if print_lattice:
        print(f"## Merged PL")
        print(M.print())
    
    ## save to file
    print(f"## Saving merged PL to GML")
    if output_gml_file and not no_gml_output:
        pattern_lattice_to_gml (M, output_gml_file, include_zscores = True, check = verbose)
        print(f"# saved to: <{output_gml_file}>")
    else:
        print(f"## Skipped GML output (use -o/--output_gml to specify file)")

    ## draw diagram of M optionally
    if not save_lattice:
        print(f"## Drawing a diagram from the merged PL")
        multibyte_font_name = setup_font ()
        M.draw_lattice (layout, MPG_key, save_lattice = save_lattice, draw_inline = draw_inline, input_name = input_file_name_stem, auto_figsizing = auto_figsizing, fig_size = fig_size, fig_dpi = fig_dpi, generality = generality, label_sample_n = label_sample_n, p_metric = p_metric, make_links_safely = make_links_safely, use_robust_zscore = use_robust_zscore, zscore_lb = zscore_lowerbound, zscore_ub = zscore_upperbound, mark_instances = mark_instances, font_name = multibyte_font_name, zscores_from_targets = zscores_from_targets, scale_factor = scale_factor, check = draw_inspection)
    else:
        print(f"## Skipped drawing (use -D/--draw-lattice to enable visualization)")

## conclude
print(f"## input_file_name: {input_file_name}")
print(f"## PL(s) built from {len(S)} sources: {[ as_label(x, sep = ',') for x in S ]}")


### end of file
