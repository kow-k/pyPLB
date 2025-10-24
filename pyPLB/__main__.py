#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#
"""
pyPLB

A Python implementation of generalized Pattern Lattice Builder (gPLB)

developed by Kow Kuroda

"Generalized" means that a pattern lattice build from [a, b, c] includes either [_, a, b, c], [a, b, c, _] and [_, a, b, c, _] (Level 1 generalization) or ['_', 'a', '_', 'b', 'c'], ['a', '_', 'b', 'c', '_'], ['_', 'a', '_', 'b', 'c', '_'], ['_', 'a', 'b', '_', 'c'], ['a', 'b', '_', 'c', '_'], ['_', 'a', 'b', '_', 'c', '_'], ['_', 'a', '_', 'b', '_', 'c'], ['a', '_', 'b', '_', 'c', '_'], ['_', 'a', '_', 'b', '_', 'c', '_'] (Level 2 generalization). Level 1 generalization is concerned with gaps at edges only, whereas Level 2 generalization with all possible insertion points. This makes pyPLB different from RubyPLB (rubyplb) developed by Yoichoro Hasebe and Kow Kuroda, available at <https://github.com/yohasebe/rubyplb>.

created on 2024/09/24
modified on
2024/09/25, 28, 29, 30; 10/01, 02, 03, 04, 05, 06, 07, 08, 09, 10, 12, 15, 16, 17, 18, 19, 20, 21, 23, 24, 30, 31; 11/01, 06, 07, 08, 09, 10, 11;

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

"""

#
## modules to use
import re
import functools
import pprint as pp
import random

## increase recursion limit
import sys
sys.setrecursionlimit(1500)

## settings
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
parser.add_argument('-c', '--input_comment_escapes', type=list, default=['#', '%'])
parser.add_argument('-d', '--input_field_seps', type=str, default=',;')
parser.add_argument('-P', '--sep2_is_suppressive', action='store_true', default=False)
parser.add_argument('-C', '--uncapitalize', action='store_true', default=False)
parser.add_argument('-H', '--split_hyphenation', action='store_false', default=True)
parser.add_argument('-X', '--remove_punctuations', action='store_false', default=True)
parser.add_argument('-g', '--gap_mark', type=str, default='_')
parser.add_argument('-t', '--tracer', type=str, default='~')
parser.add_argument('-m', '--max_size', type=int, default=None)
parser.add_argument('-s', '--sample_n', type=int, default=None)
parser.add_argument('-D', '--add_displaced_versions', action='store_true', default=False)
parser.add_argument('-R', '--unreflexive', action='store_false', default=True)
parser.add_argument('-G', '--generality', type=int, default=0)
parser.add_argument('-p', '--productivity_metric', type=str, default='rank')
parser.add_argument('-z', '-zl', '--zscore_lowerbound', type=float, default=None)
parser.add_argument('-zu', '--zscore_upperbound', type=float, default= None)
parser.add_argument('-Z', '--use_robust_zscore', action='store_false', default=True)
parser.add_argument('-T', '--zscores_from_targets', action='store_true', default=False)
parser.add_argument('-A', '--auto_figsizing', action='store_true', default=False)
parser.add_argument('-E', '--scaling_factor', type= float, default=5)
parser.add_argument('-F', '--fig_size', type=parse_tuple_for_arg, default=(10,9))
parser.add_argument('-I', '--draw_individual_lattices', action='store_true', default=False)
parser.add_argument('-L', '--layout', type= str, default= 'Multi_partite')
parser.add_argument('-J', '--use_multibyte_chars', action='store_true', default=False)
parser.add_argument('-K', '--MPG_key', type=str, default='gap_size')
#parser.add_argument('-S', '--sample_id', type= int, default= 1)
parser.add_argument('-S', '--build_lattice_stepwise', action='store_true', default=False)
parser.add_argument('-i', '--mark_instances', action='store_true', default=False)
parser.add_argument('-M', '--use_mp', action='store_false', default=True)
parser.add_argument('-N', '--print_link_targets', action='store_true', default=False)
parser.add_argument('-o', '--print_forms', action='store_true', default=False)
parser.add_argument('-Y', '--phrasal', action='store_true', default=False)

##
args = parser.parse_args()
##
file                   = args.file   # process a file when it exists
verbose                = args.verbose
detailed               = args.detailed
use_mp                 = args.use_mp # controls use of multiprocess
input_comment_escapes  = args.input_comment_escapes
input_field_seps       = args.input_field_seps
sep2_is_suppressive    = args.sep2_is_suppressive # controls the behavior of second sep
uncapitalize           = args.uncapitalize
remove_punct           = args.remove_punctuations
split_hyphenation      = args.split_hyphenation
gap_mark               = args.gap_mark
tracer                 = args.tracer
max_size               = args.max_size
#sample_id              = args.sample_id
sample_n               = args.sample_n
reflexive              = args.unreflexive
generality             = args.generality
add_displaced_versions = args.add_displaced_versions
build_lattice_stepwise = args.build_lattice_stepwise
print_link_targets     = args.print_link_targets
layout                 = args.layout
MPG_key                = args.MPG_key
p_metric               = args.productivity_metric
auto_figsizing         = args.auto_figsizing
fig_size               = args.fig_size
zscore_lowerbound      = args.zscore_lowerbound
zscore_upperbound      = args.zscore_upperbound
use_robust_zscore      = args.use_robust_zscore
zscores_from_targets   = args.zscores_from_targets
mark_instances         = args.mark_instances
draw_individually      = args.draw_individual_lattices
use_multibyte_chars    = args.use_multibyte_chars
scale_factor           = args.scaling_factor
print_forms            = args.print_forms
phrasal                = args.phrasal

## inspection paramters
draw_inspection      = False

## show paramters
print(f"##Parameters")
print(f"#use_multiprocess: {use_mp}")
print(f"#detailed: {detailed}")
print(f"#verbose: {verbose}")
print(f"#input_comment_escapes: {input_comment_escapes}")
print(f"#input_field_seps: {input_field_seps}")
print(f"#sep2_is_suppressive: {sep2_is_suppressive}")
print(f"#uncapitalize: {uncapitalize}")
print(f"#remove_punctuations: {remove_punct}")
print(f"#split_hyphenation: {split_hyphenation}")
print(f"#gap_mark: {gap_mark}")
print(f"#instantiation is reflexive: {reflexive}")
print(f"#building lattice with generality: {generality}")
print(f"#p_metric [productivity metric]: {p_metric}")
print(f"#zscores_from_targets: {zscores_from_targets}")
print(f"#use_robust_zscore: {use_robust_zscore}")
print(f"#zscore_lowerbound: {zscore_lowerbound}")
print(f"#zscore_upperbound: {zscore_upperbound}")
print(f"#mark_instances: {mark_instances}")
print(f"#auto_figsizing: {auto_figsizing}")
print(f"#fig_size: {fig_size}")
print(f"#draw_individually: {draw_individually}")

### Functions
##
def segment_with_levels (lines: list, seps: str, sep2_is_suppressive: bool, remove_punct: bool, split_hyphenation: bool, uncapitalize: bool) -> list:

    assert len(lines) > 0
    sep_list = list(seps)
    assert len(sep_list) > 0
    if sep2_is_suppressive:
        ignored_sep, primary_sep, *_ = sep_list
        #lines = [ line.replace(ignored_sep, "").split(primary_sep) for line in lines ]
        ## The line above fails
        lines = [ f"{line}{primary_sep}".replace(ignored_sep, "").split(primary_sep) for line in lines ]
    else:
        print(f"#seps: {seps}")
        lines = [ re.split(f"[{seps}]", line) for line in lines ]

    ## remove punctuations from lines
    punct_symbols = list(",.?!:;/\–~")
    if remove_punct:
        lines = [ [ x for x in line if x not in punct_symbols ] for line in lines ]

    ## split hyphenated tokens
    if split_hyphenation:
        lines = [ process_hyphenation (line) for line in lines ]

    ## uncapitalize tokens over lines
    if uncapitalize:
        lines = [ [ x.lower() for x in line ] for line in lines ]

    ##
    return lines

##
def parse_input (file, comment_escapes: list, field_seps: str, remove_punct: bool = remove_punct, split_hyphenation: bool = split_hyphenation, uncapitalize: bool = uncapitalize, check: bool = False) -> list:
    """
    reads a file, splits it into segments using a given separator, removes comments, and forward the result to main
    """

    ## reading data
    with file as f:
        lines =  [ line.strip() for line in f.readlines() if not line[0] in comment_escapes ]
    if check:
        print(f"#input: {lines}")

    ## remove inline comments
    filtered_lines = [ ]
    for line in lines:
        filtered_line = []
        for char in line:
            if char not in comment_escapes:
                filtered_line.append(char)
            else:
                break
        filtered_lines.append("".join(filtered_line))
    ##
    if check:
        print(f"#filtered_lines: {filtered_lines}")

    ## generate segmentations
    segmented_lines = segment_with_levels (filtered_lines, seps = field_seps, sep2_is_suppressive = sep2_is_suppressive, remove_punct = remove_punct, split_hyphenation = split_hyphenation, uncapitalize = uncapitalize)

    ##
    return segmented_lines

## set font for Japanese character display
import matplotlib
if use_multibyte_chars:
    from matplotlib import font_manager as Font_manager
    ## select font
    multibyte_font_names = [    "IPAexGothic",  # 0 Multi-platform font
                                "Hiragino sans" # 1 Mac only
                            ]
    multibyte_font_name  = multibyte_font_names[0]

    ## tell where target fonts are
    system_font_dir = "/System/Library/Fonts/"
    user_font_dir = "/Library/Fonts/"

    # use the version installed via TeXLive
    user_font_dir2 = "/usr/local/texlive/2013/texmf-dist/fonts/truetype/public/ipaex/"
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
print(f"multibyte_font_name: {multibyte_font_name}")
print(f"matplotlib.rcParams['font.family']: {matplotlib.rcParams['font.family']}")

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

## process
S0 = []
if not file is None:
    input_parses = parse_input (file, comment_escapes = input_comment_escapes, field_seps = input_field_seps, remove_punct = remove_punct, split_hyphenation = split_hyphenation, uncapitalize = uncapitalize, check = False)
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
            raise f"sample_id {sample_id} is not defined"
        sample_S = [ [ seg for seg in re.split(r"", t) if len(seg) > 0 ] for t in Words ]
        S0.append (sample_S)

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
    print(f"#S: {S}")

## select source
print(f"##Source lists:")
for i, s in enumerate(S):
    print (f"#source {i}: {s}")


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
        print(f"#processing: {s}")
        try:
            p = Pattern(s, gap_mark = gap_mark)
        except TypeError:
            p = Pattern(s, gap_mark = gap_mark, tracer = tracer)
        if detailed:
            print(f"#p: {p}")
        Patterns.append(p)
##
Patterns = sorted (Patterns, key = lambda x: len(x), reverse = False)
##
for i, pat in enumerate(Patterns):
    if verbose:
        print(f"#gapped patterns from pattern {i}: {pat}")
    for i, g_pat in enumerate(pat.create_gapped_versions (check = False)):
        if verbose:
            print(f"# gapped {i+1}: {g_pat}")
#exit()

##
print(f"##Generating g{generality}PLs ...")
L = [ ]
for i, p in enumerate(Patterns):
    print(f"#generating g{generality}PL {i+1} from {p}")
    ## main
    patlat = PatternLattice (p, generality = generality, reflexive = reflexive, check = False)
    if detailed:
        pp.pprint(patlat)
    ##
    if verbose:
        print(f"#patlat.origin: {patlat.origin}")
        if detailed:
            pp.pprint (patlat.origin)
    ##
    if verbose:
        print(f"#patlat.nodes; count: {len(patlat.nodes)}")
        if detailed:
            pp.pprint (patlat.nodes)
    ##
    if verbose:
        print(f"#patlat.ranked_nodes; count: {len(patlat.ranked_nodes)}")
        if detailed:
            pp.pprint (patlat.ranked_nodes)
    ##
    if verbose:
        print(f"#patlat.links; count: {len(patlat.links)}")
        if detailed:
            pp.pprint (patlat.links)
    ##
    L.append (patlat)
#exit()

##
if detailed:
    for i, patlat in enumerate(L):
        for j, pattern in enumerate(patlat):
            print(f"#p{i}.{j}: {pattern}")
#exit()

## print forms and then quit without drawing lattices
if print_forms:
    joint = input_field_sep
    for i, patlat in enumerate(L):
        for j, pat in enumerate(patlat):
            print(f"p{i:02d}.form{j:03d}: {joint.join(pat.get_form())}")
    exit()

## draw lattices and then quit without drawing the merged lattice
if draw_individually:
    print(f"##Drawing g{generality}PLs individually")
    for i, patlat in enumerate(L):
        print(f"#Drawing a diagram from g{generality}PL {i+1}")
        patlat.draw_network (layout = layout, MPG_key = MPG_key, auto_figsizing = auto_figsizing, fig_size = fig_size, generality = generality, p_metric = p_metric, zscores_from_targets = zscores_from_targets, mark_instances = mark_instances, scale_factor = scale_factor, font_name = multibyte_font_name, check = draw_inspection)
    exit()

##
print(f"##Merging {len(L)} g{generality}PLs ...")
simplified     = False
label_sample_n = 5
if simplified:
    #print(f"#binary merger")
    La, Lb = L[0], L[1]
    if verbose:
        print(f"#La: {La}")
        print(f"#Lb: {Lb}")
    M = La.merge_with (Lb, use_mp = use_mp, show_steps = True, check = False)

## Individual draw
elif build_lattice_stepwise:
    gen_links_internally = True
    print(f"##Mergig g{generality}PLs ...")
    for i, patlat in enumerate (L):
        print(f"#Processing g{generality}PL {i+1}")
        if i == 0:
            M = patlat
        else: ## merger
            M = M.merge_with (patlat, gen_links_internally = gen_links_internally, use_mp = use_mp, generality = generality, reflexive = reflexive, reductive = True, show_steps = True, check = False)
            ## delete the original
            patplat = None

        ## check nodes in M
        print(f"merged g{generality}PL with {len(M.nodes)} nodes")
        for i, p in enumerate(M.nodes):
            print(f"#node {i:3d}: {p.separate_print()}")

        ## genenrate links in delay
        if len(M.links) == 0 and not gen_links_internally:
            ## Don't do: M = M.update(...)
            M.update_links (p_metric, reflexive = reflexive, check = False) ## Crucially
        print(f"#generated {len(M.links)} links")

        ## checking links in M
        print(f"##Links")
        for i, link in enumerate(M.links):
            link.pprint (indicator = i, paired = True, link_type = "instantiates", check = False)

        ## generate z-scores from link targets
        gen_zscores_from_targets_by (p_metric, M, gap_mark = gap_mark, use_robust_zscore = use_robust_zscore, check = False)
        if print_link_targets:
            for node, zscore in M.source_zscores.items():
                print(f"#node {node} has z-score {zscore: .3f}")

        ## generate z-scores from link sources
        gen_zscores_from_sources_by (p_metric, M, gap_mark = gap_mark, use_robust_zscore = use_robust_zscore, check = False)
        for node, zscore in M.source_zscores.items():
            print(f"#node {node} has z-score {zscore: .3f}")
        ##
        print(f"##Results")
        M.draw_network (layout, MPG_key, auto_figsizing = auto_figsizing, fig_size = fig_size, generality = generality, label_sample_n = label_sample_n, p_metric = p_metric, use_robust_zscore = use_robust_zscore, zscore_lb = zscore_lowerbound, zscore_ub = zscore_upperbound, mark_instances = mark_instances, font_name = multibyte_font_name, zscores_from_targets = zscores_from_targets, scale_factor = scale_factor, check = draw_inspection)
## Draw after integration
else:
    gen_links_internally = False
    M = functools.reduce (lambda La, Lb: La.merge_with (Lb, gen_links_internally = gen_links_internally, use_mp = use_mp, generality = generality, reflexive = reflexive, reductive = True, check = False), L)

    # The following process was isolated for memory conservation
    if len(M.links) == 0 and not gen_links_internally:
        print(f"##Generating links independently")
        ## Don't do: M = M.update(...)
        M.update_links (p_metric, reflexive = reflexive, use_mp = use_mp, check = False)

    ##
    print(f"##Results")

    ## check nodes in M
    print(f"#Merger has {len(M.nodes)} nodes")
    for i, p in enumerate(M.nodes):
        print(f"#node {i}: {p}")

    ## check links in M
    print(f"#Merger has {len(M.links)} links")
    for i, link in enumerate(M.links):
        link.pprint (indicator = i, paired = True, link_type = "instantiates", check = False)

    ## get z-scores from link targets
    print(f"##Calculating z-scores ...")
    gen_zscores_from_targets_by (p_metric, M, gap_mark = gap_mark, tracer = tracer, use_robust_zscore = use_robust_zscore, check = False)
    if print_link_targets:
        for node, zscore in M.target_zscores.items():
            print(f"#node {node} has z-score {zscore: .3f} [n: {M.link_targets[node]:2d}]")

    ## get z-scores from link sources
    gen_zscores_from_sources_by (p_metric, M, gap_mark = gap_mark, tracer = tracer, use_robust_zscore = use_robust_zscore, check = False)
    for node, zscore in M.source_zscores.items():
        print(f"#node {node} has z-score {zscore: .3f} (n: {M.link_sources[node]:2d})")

    ## draw diagram of M
    print(f"##Drawing a diagram from the merged PL")
    M.draw_network (layout, MPG_key, auto_figsizing = auto_figsizing, fig_size = fig_size, generality = generality, label_sample_n = label_sample_n, p_metric = p_metric, use_robust_zscore = use_robust_zscore, zscore_lb = zscore_lowerbound, zscore_ub = zscore_upperbound, mark_instances = mark_instances, font_name = multibyte_font_name, zscores_from_targets = zscores_from_targets, scale_factor = scale_factor, check = draw_inspection)

## conclude
print(f"##built from {len(S)} sources: {[ as_label(x, sep = ',') for x in S ]}")


### end of file
