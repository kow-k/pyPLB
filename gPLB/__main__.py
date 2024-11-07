#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#
"""
gPLB.py

A Python implementation of generalized Pattern Lattice Builder (gPLB)

developed by Kow Kuroda

"Generalized" means that a pattern lattice build from [a, b, c] includes [_, a, b, c], [a, b, c, _] and [_, a, b, c, _]. This makes gPLB different from RubyPLB (rubyplb) developed by Yoichoro Hasebe and Kow Kuroda, available at <https://github.com/yohasebe/rubyplb>.

created on 2024/09/24
modified on 2024/09/25, 28, 29, 30; 10/01, 02, 03, 04, 05, 06, 07, 08, 09, 10, 12, 15, 16, 17, 18, 19, 20, 21, 23, 24, 30, 31; 11/01

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
from pattern_link import *
from pattern_lattice import *

## settings
import argparse
parser  = argparse.ArgumentParser(description = "")
parser.add_argument('file', type= open, default= None)
parser.add_argument('-P', '--phrasal', action= 'store_true', default= False)
parser.add_argument('-v', '--verbose', action= 'store_true', default= False)
parser.add_argument('-w', '--detailed', action= 'store_true', default= False)
parser.add_argument('-f', '--input_field_sep', type= str, default= ',')
parser.add_argument('-c', '--input_comment_escape', type= str, default= '#')
parser.add_argument('-g', '--gap_mark', type= str, default= '_')
parser.add_argument('-R', '--unreflexive', action= 'store_false', default= True)
parser.add_argument('-G', '--generalized', action= 'store_false', default= True)
parser.add_argument('-m', '--max_size', type= int, default= None)
parser.add_argument('-n', '--sample_n', type= int, default= None)
#parser.add_argument('-S', '--sample_id', type= int, default= 1)
parser.add_argument('-S', '--build_lattice_stepwise', action= 'store_true', default= False)
parser.add_argument('-F', '--scaling_factor', type= float, default= 5)
parser.add_argument('-z', '-zl', '--zscore_lowerbound', type= float, default= None)
parser.add_argument('-zu', '--zscore_upperbound', type= float, default= None)
parser.add_argument('-Z', '--use_robust_zscore', action='store_true', default= False)
parser.add_argument('-T', '--zscores_from_targets', action='store_true', default= False)
parser.add_argument('-t', '--print_link_targets', action='store_true', default= False)
parser.add_argument('-D', '--draw_stepwise', action= 'store_false', default = True)
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
input_field_sep         = args.input_field_sep
input_comment_escape    = args.input_comment_escape
gap_mark                = args.gap_mark
max_size                = args.max_size
#sample_id               = args.sample_id
sample_n                = args.sample_n
generalized             = args.generalized
reflexive               = args.unreflexive
build_lattice_stepwise  = args.build_lattice_stepwise
draw_stepwise           = args.draw_stepwise
layout                  = args.layout
auto_fig_sizing         = args.auto_fig_sizing
zscore_lowerbound       = args.zscore_lowerbound
zscore_upperbound       = args.zscore_upperbound
use_robust_zscore       = args.use_robust_zscore
zscores_from_targets    = args.zscores_from_targets
print_link_targets      = args.print_link_targets
scale_factor            = args.scaling_factor
use_multibyte_chars     = args.use_multibyte_chars

### implications
# diagram drawing
if not layout is None:
    draw_stepwise       = True
## z-scores handling
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
print(f"#lattice is generalized: {generalized}")
print(f"#instantiation is reflexive: {reflexive}")
print(f"#gap_mark: {gap_mark}")
print(f"#draw_stepwise: {draw_stepwise}")
print(f"#use_robust_zscore: {use_robust_zscore}")
print(f"#zscore_lowerbound: {zscore_lowerbound}")
print(f"#zscore_upperbound: {zscore_upperbound}")
print(f"#zscores_from_targets: {zscores_from_targets}")
print(f"#mp_inspection: {mp_inspection}")

### Functions

##
def parse_input (file, field_sep: str = ",", comment_escape: str = "#") -> None:
    "reads a file, splits it into segments using a given separator, removes comments, and forward the result to main"
    import csv
    ## reading data
    data = list(csv.reader (file, delimiter = field_sep)) # Crucially list(..)

    ## discard comment lines that start with #
    data = [ [ x.strip() for x in F ] for F in data if len(F) > 0 and not F[0][0] == comment_escape ]

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
    S0 = parse_input (file, field_sep = input_field_sep, comment_escape = input_comment_escape)
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
if sample_n is not None:
    S = random.sample (S0, sample_n)
else:
    S = S0
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
## check font settings
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
print(f"##Generating (generalized) PatternLattices ...")
L = [ ]
for i, p in enumerate(Patterns):
    print(f"#generating PatternLattice {i+1} from {p}")
    ## main
    patlat = PatternLattice (p, generalized = generalized, reflexive = reflexive, check = False)
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
if draw_stepwise and verbose:
    print(f"##Drawing diagrams")
    for i, patlat in enumerate(L):
        print(f"#drawing diagram from PatternLattice {i+1}")
        patlat.draw_diagrams (layout = layout, generalized = generalized, auto_fig_sizing = auto_fig_sizing, scale_factor = scale_factor, font_name = multibyte_font_name, check = draw_inspection)
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

elif build_lattice_stepwise:
    label_sample_n = 10
    #null_pat = Pattern([], gap_mark = gap_mark)
    #M = PatternLattice(null_pat, generalized = generalized, check = False)
    gen_links_internally = True
    M = L[0]
    for patlat in L[1:]:
        M = M.merge_lattices (patlat, gen_links_internally = gen_links_internally, use_multiprocess = use_mp, reflexive = reflexive, show_steps = True, check = False)

        ## check nodes in M
        print(f"generated {len(M.nodes)} Patterns")
        if verbose:
            print(f"#Patterns")
            for i, p in enumerate(M.nodes):
                print(f"#Pattern {i+1}: {p}")

        ##
        if not gen_links_internally and len(M.links) > 0:
            M.update_links (reflexive = reflexive, check = False) ## Crucially

        ## checking links in M
        print(f"##Links")
        print(f"#generated {len(M.links)} links")
        for i, link in enumerate(M.links):
            link.pprint (indicator = i, paired = True, link_type = "instantiates", check = False)

        ## generate z-scores from link targets
        gen_zscores_from_targets (M, gap_mark = gap_mark, use_robust_zscore = use_robust_zscore, check = False)

        ## generate z-scores from link sources
        gen_zscores_from_sources (M, gap_mark = gap_mark, use_robust_zscore = use_robust_zscore, check = False)
        ##
        print(f"##Results")
        if draw_stepwise:
            M.draw_diagrams (layout = layout, generalized = generalized, auto_fig_sizing = auto_fig_sizing, label_sample_n = label_sample_n, use_robust_zscore = use_robust_zscore, zscore_lowerbound = zscore_lowerbound, zscore_upperbound = zscore_upperbound, font_name = multibyte_font_name, zscores_from_sources = zscores_from_sources, scale_factor = scale_factor, check = draw_inspection)

else:
    gen_links_internally = False
    M = functools.reduce (lambda La, Lb: La.merge_lattices (Lb, gen_links_internally = gen_links_internally, use_multiprocess = use_mp, reflexive = reflexive, show_steps = True, check = False), L)
    # The following process was isolated for memory conservation
    if not gen_links_internally and len(M.links) == 0:
        print(f"##Generating links independently")
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
    
    ## get z-scores from link targets
    gen_zscores_from_targets (M, gap_mark = gap_mark, use_robust_zscore = use_robust_zscore, check = False)
    if print_link_targets:
        print(M.target_zscores)
    
    ## get z-scores from link sources
    gen_zscores_from_sources (M, gap_mark = gap_mark, use_robust_zscore = use_robust_zscore, check = False)
    print(M.source_zscores)
    
    ## draw diagram of M
    print(f"##Drawing a diagram from the merged lattice")
    label_sample_n = 10
    M.draw_diagrams (layout = layout, generalized = generalized, auto_fig_sizing = auto_fig_sizing, label_sample_n = label_sample_n, use_robust_zscore = use_robust_zscore, zscore_lowerbound = zscore_lowerbound, zscore_upperbound = zscore_upperbound, font_name = multibyte_font_name, zscores_from_sources = zscores_from_sources, scale_factor = scale_factor, check = draw_inspection)

## conclude
print(f"##built from {len(S)} sources: {[ as_label(x, sep = ',') for x in S ]}")

### end of file
