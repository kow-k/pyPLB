#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#
"""
gPLB.py

A Python implementation of generalized Pattern Lattice Builder (gPLB)

developed by Kow Kuroda

"Generalized" means that a pattern lattice build from [a, b, c] includes [_, a, b, c], [a, b, c, _] and [_, a, b, c, _]. This makes gPLB different from RubyPLB (rubyplb) developed by Yoichoro Hasebe and Kow Kuroda, available at <https://github.com/yohasebe/rubyplb>.

created on 2024/09/24
modified on 2024/09/25, 28, 29, 30; 10/01, 02, 03, 04, 05, 06, 07, 08, 09, 10, 12, 15, 16, 17, 18, 19, 20, 21, 23, 24, 30, 31

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
"""

#
## modules to use
import re
import functools
import pprint as pp
import random

#import multiprocessing as mp
##
from utils import *
from pattern import *
from pattern import *
from pattern_lattice import *

## settings
import argparse
parser  = argparse.ArgumentParser(description = "")
parser.add_argument('file', type= open, default= None)
parser.add_argument('-P', '--phrasal', action= 'store_true', default= False)
parser.add_argument('-v', '--verbose', action= 'store_true', default= False)
parser.add_argument('-w', '--detailed', action= 'store_true', default= False)
parser.add_argument('-s', '--input_field_sep', type= str, default= ',')
parser.add_argument('-c', '--input_comment_escape', type= str, default= '#')
parser.add_argument('-g', '--gap_mark', type= str, default= '_')
parser.add_argument('-R', '--unreflexive', action= 'store_false', default= True)
parser.add_argument('-G', '--generalized', action= 'store_false', default= True)
parser.add_argument('-m', '--max_size', type= int, default= None)
parser.add_argument('-n', '--sample_n', type= int, default= 3)
parser.add_argument('-S', '--sample_id', type= int, default= 1)
parser.add_argument('-F', '--scaling_factor', type= float, default= 5)
parser.add_argument('-z', '--zscore_lowerbound', type= float, default= None)
parser.add_argument('-Z', '--use_robust_zscore', action='store_true', default= False)
parser.add_argument('-T', '--zscores_from_targets', action='store_true', default= False)
parser.add_argument('-D', '--draw_diagrams', action= 'store_false', default = True)
parser.add_argument('-J', '--use_multibyte_chars', action= 'store_true', default = False)
parser.add_argument('-L', '--layout', type= str, default= 'Multi_partite')
parser.add_argument('-A', '--auto_fig_sizing', action= 'store_true', default= False)
##
args = parser.parse_args()
##
file                    = args.file   # process a file when it exists
phrasal                 = args.phrasal
verbose                 = args.verbose
detailed                = args.detailed
max_size                = args.max_size
sample_id               = args.sample_id
sample_n                = args.sample_n
generalized             = args.generalized
gap_mark                = args.gap_mark
input_field_sep         = args.input_field_sep
input_comment_escape    = args.input_comment_escape
reflexive               = args.unreflexive
draw_diagrams           = args.draw_diagrams
layout                  = args.layout
auto_fig_sizing         = args.auto_fig_sizing
zscore_lowerbound       = args.zscore_lowerbound
use_robust_zscore       = args.use_robust_zscore
zscores_from_targets    = args.zscores_from_targets
scale_factor            = args.scaling_factor
use_multibyte_chars     = args.use_multibyte_chars

## implications
if not layout is None:
    draw_diagrams       = True
zscores_from_sources    = not zscores_from_targets

## inspection paramters
draw_inspection      = False
mp_inspection        = False # This disables use of multiprocess
if mp_inspection:
    use_mp = False
else:
    use_mp = True

## show paramters
print(f"##Parameters")
print(f"#verbose: {verbose}")
print(f"#detailed: {detailed}")
print(f"#input_field_sep: {input_field_sep}")
print(f"#input_comment_escape: {input_comment_escape}")
print(f"#gap_mark: {gap_mark}")
print(f"#lattice is generalized: {generalized}")
print(f"#instantiation is reflexive: {reflexive}")
print(f"#use_robust_zscore: {use_robust_zscore}")
print(f"#zscores_from_targets: {zscores_from_targets}")
print(f"#draw_diagrams: {draw_diagrams}")
print(f"#mp_inspection: {mp_inspection}")

### Functions

##
def parse_input (file, field_sep: str = ",", comment_escape: str = "#") -> None:
    "reads a file, splits it into segments using a given separator, removes comments, and forward the result to main"
    import csv
    ## reading data
    data = list(csv.reader (file, delimiter = field_sep)) # Crucially list(..)

    ## discard comment lines that start with #
    data = [ F for F in data if len(F) > 0 and not F[0][0] == comment_escape ]

    ## remove in-line comments
    data_renewed = [ ]
    for F in data:
        G = []
        for f in F:
            pos = f.find (comment_escape)
            if pos > 0:
                G.append(f[:pos])
                continue
            else:
                G.append(f)
        ##
        data_renewed.append(G)
    ##
    return data_renewed

## process
if not file is None:
    S0 = parse_input(file, field_sep = input_field_sep, comment_escape = input_comment_escape)
else:
    ## phrasal source
    if phrasal:
        Text1 = [ 'a big boy', 'the big boy', 'a big girl', 'the big girl',
            'a funny boy', 'the funny boy', 'the funny boys', 'funny boys',
            'a small boy', 'a small girl', 'the small boy', 'the small girl',
            'big boys', 'big girls', 'small boys', 'small girls', 'the funny girl',
            'the big boys', 'the small boys', 'the boys', 'the girls' ]
        Phrases1 = [ t.split() for t in Text1 ]
        S0 = Phrases1
    ## lexical sources
    else:
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
        S0 = [ [ seg for seg in re.split(r"", t) if len(seg) > 0 ] for t in Words ]

## filter
if not max_size is None:
    S0 = [ x for x in S0 if len(x) <= max_size and len(x) > 0 ]

## take a sample
S = random.sample (S0, sample_n)
if verbose:
    print(f"# S: {S}")

## select source
print(f"##Source lists:")
for i, s in enumerate(S):
    print (f"#source {i}: {s}")

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
##
print(f"multibyte_font_name: {multibyte_font_name}")
print(f"matplotlib.rcParams['font.family']: {matplotlib.rcParams['font.family']}")
## generating patterns
Patterns = [ ]
for s in S:
    if verbose:
        print(f"#processing: {s}")
    p = Pattern(s, gap_mark = gap_mark)
    if detailed:
        print(f"#p: {p}")
    Patterns.append(p)

##
for i, pat in enumerate(Patterns):
    if verbose:
        print(f"#gapped patterns from pattern {i}: {pat}")
    for i, g_pat in enumerate(pat.create_gapped_versions (check = False)):
        if verbose:
            print(f"# gapped {i+1}: {g_pat}")
##
#exit()
##
print(f"##Generating PatternLattices ...")
L = [ ]
for i, p in enumerate(Patterns):
    print(f"#generating PatternLattice {i+1} from {p}")
    ## main
    patlat = PatternLattice(p, reflexive = reflexive, generalized = generalized, check = False)
    if detailed:
        pp.pprint(patlat)
    ##
    if verbose:
        print(f"#patlat.origin: {patlat.origin}")
        if detailed:
            pp.pprint(patlat.origin)
    ##
    if verbose:
        print(f"#patlat.nodes; count: {len(patlat.nodes)}")
        if detailed:
            pp.pprint(patlat.nodes)
    ##
    if verbose:
        print(f"#patlat.ranked_nodes; count: {len(patlat.ranked_nodes)}")
        if detailed:
            pp.pprint(patlat.ranked_nodes)
    ##
    if verbose:
        print(f"#patlat.links; count: {len(patlat.links)}")
        if detailed:
            pp.pprint(patlat.links)
    ##
    L.append(patlat)
##
#exit()
##
if detailed:
    for i, patlat in enumerate(L):
        for j, pattern in enumerate(patlat):
            print(f"#p{i}.{j}: {pattern}")
##
#exit()
##
if draw_diagrams and verbose:
    print(f"##Drawing diagrams")
    for i, patlat in enumerate(L):
        print(f"#drawing diagram from: PatternLattice {i+1}")
        patlat.draw_diagrams (layout = layout, font_name = multibyte_font_name, auto_fig_sizing = auto_fig_sizing, scale_factor = scale_factor, check = draw_inspection)
##
#exit()
##
print(f"##Merging {len(L)} PatternLattices ...")

simplified = False
if simplified:
    #print(f"#binary merger")
    La, Lb = L[0], L[1]
    if verbose:
        print(f"#La: {La}")
        print(f"#Lb: {Lb}")
    M = La.merge_lattices (Lb, show_steps = True, check = False)
else:
    gen_links_internally = False
    M = functools.reduce (lambda La, Lb: La.merge_lattices (Lb, gen_links = gen_links_internally, use_multiprocess = use_mp, reflexive = reflexive, show_steps = True, check = False), L)
    # The following process was isolated for speeding up
    if len(M.links) == 0 and not gen_links_internally:
        print(f"#Generating links independently")
        M.update_links (reflexive = reflexive, check = False)

##
print(f"##Results")

## check nodes in M
print(f"generated {len(M.nodes)} Patterns")
if verbose:
    print(f"#Patterns")
    for i, p in enumerate(M.nodes):
        print(f"#Pattern {i+1}: {p}")

## checking links in M
print(f"##Links")
print(f"#generated {len(M.links)} links")
for i, link in enumerate(M.links):
    link.pprint (indicator = i, paired = True, link_type = "instantiates", check = False)

## adding link source z-scores to M
if verbose:
    print(f"##Link_sources")
Link_sources     = M.link_sources
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
    print(f"#source {i:3d}: {link_source} has {value} link(s) [{source_zscores[link_source]: .5f} at rank {rank}]")

## attach source_zscores to M
#M.source_zscores = source_zscores
M.source_zscores.update(source_zscores)
if verbose:
    print(f"M.source_zscores: {M.source_zscores}")


## adding link target z-scores to M
if verbose:
    print(f"##Link_targets")
Link_targets     = M.link_targets
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
    print(f"#target {i:3d}: {link_target} has {value} link(s) [{target_zscores[link_target]: .5f} at rank {rank}]")

## attach source_zscores to M
#M.source_zscores = source_zscores
M.target_zscores.update(target_zscores)
if verbose:
    print(f"M.target_zscores: {M.target_zscores}")

## draw diagram of M
if draw_diagrams:
    print(f"#Drawing a diagram from the merged lattice")
    M.draw_diagrams (layout = layout, auto_fig_sizing = auto_fig_sizing, zscore_lowerbound = zscore_lowerbound, font_name = multibyte_font_name, zscores_from_sources = zscores_from_sources, scale_factor = scale_factor, check = draw_inspection)

### end of file
