#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#
"""
gPLB.py

A Python implementation of generalized Pattern Lattice Builder (gPLB)

developed by Kow Kuroda

"Generalized" means that a pattern lattice build from [a, b, c] includes [_, a, b, c], [a, b, c, _] and [_, a, b, c, _]. This makes gPLB different from RubyPLB (rubyplb) developed by Yoichoro Hasebe and Kow Kuroda, available at <https://github.com/yohasebe/rubyplb>.

created on 2024/09/24
modified on 2024/09/25, 28, 29, 30; 10/01, 02, 03, 04, 05, 06, 07, 08, 09, 10, 12, 15, 16, 17, 18, 19, 20, 21, 23, 24, 30, 31; 11/01, 06, 07, 08, 09, 10, 11

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
parser.add_argument('-v', '--verbose', action= 'store_true', default= False)
parser.add_argument('-w', '--detailed', action= 'store_true', default= False)
parser.add_argument('-f', '--input_field_sep', type= str, default= ',')
parser.add_argument('-c', '--input_comment_escapes', type= list, default= ['#', '%'])
parser.add_argument('-X', '--phrasal', action= 'store_true', default= False)
parser.add_argument('-C', '--uncapitalize', action='store_false', default= True)
parser.add_argument('-P', '--remove_punctuations', action='store_false', default= True)
parser.add_argument('-H', '--split_hyphenation', action='store_false', default= True)
parser.add_argument('-g', '--gap_mark', type= str, default= '_')
parser.add_argument('-R', '--unreflexive', action= 'store_false', default= True)
parser.add_argument('-G', '--generalized', action= 'store_false', default= True)
parser.add_argument('-m', '--max_size', type= int, default= None)
parser.add_argument('-n', '--sample_n', type= int, default= None)
#parser.add_argument('-S', '--sample_id', type= int, default= 1)
parser.add_argument('-A', '--no_auto_figsize_adjust', action='store_true', default= False)
parser.add_argument('-S', '--build_lattice_stepwise', action= 'store_true', default= False)
parser.add_argument('-I', '--draw_individual_lattices', action= 'store_true', default = False)
parser.add_argument('-F', '--scaling_factor', type= float, default= 5)
parser.add_argument('-z', '-zl', '--zscore_lowerbound', type= float, default= None)
parser.add_argument('-zu', '--zscore_upperbound', type= float, default= None)
parser.add_argument('-Z', '--use_robust_zscore', action='store_true', default= False)
parser.add_argument('-T', '--zscores_from_targets', action='store_true', default= False)
parser.add_argument('-t', '--print_link_targets', action='store_true', default= False)
parser.add_argument('-J', '--use_multibyte_chars', action= 'store_true', default = False)
parser.add_argument('-L', '--layout', type= str, default= 'Multi_partite')
parser.add_argument('-o', '--print_forms', action='store_true', default= False)

##
args = parser.parse_args()
##
file                    = args.file   # process a file when it exists
verbose                 = args.verbose
detailed                = args.detailed
input_comment_escapes   = args.input_comment_escapes
input_field_sep         = args.input_field_sep
phrasal                 = args.phrasal
uncapitalize            = args.uncapitalize
remove_punctuations     = args.remove_punctuations
split_hyphenation       = args.split_hyphenation
gap_mark                = args.gap_mark
max_size                = args.max_size
#sample_id               = args.sample_id
sample_n                = args.sample_n
generalized             = args.generalized
reflexive               = args.unreflexive
build_lattice_stepwise  = args.build_lattice_stepwise
print_link_targets      = args.print_link_targets
no_auto_figsize_adjust  = args.no_auto_figsize_adjust
layout                  = args.layout
draw_individually       = args.draw_individual_lattices
print_forms             = args.print_forms
scale_factor            = args.scaling_factor
use_multibyte_chars     = args.use_multibyte_chars
zscore_lowerbound       = args.zscore_lowerbound
zscore_upperbound       = args.zscore_upperbound
use_robust_zscore       = args.use_robust_zscore
zscores_from_targets    = args.zscores_from_targets

### implications
## inspection paramters
draw_inspection      = False
mp_inspection        = False # This disables use of multiprocess
if mp_inspection:
    use_mp = False
else:
    use_mp = True

## show paramters
print(f"##Parameters")
print(f"#mp_inspection: {mp_inspection}")
print(f"#detailed: {detailed}")
print(f"#verbose: {verbose}")
print(f"#input_field_sep: {input_field_sep}")
print(f"#input_comment_escapes: {input_comment_escapes}")
print(f"#uncapitalize: {uncapitalize}")
print(f"#remove_punctuations: {remove_punctuations}")
print(f"#split_hyphenation: {split_hyphenation}")
print(f"#lattice is generalized: {generalized}")
print(f"#instantiation is reflexive: {reflexive}")
print(f"#gap_mark: {gap_mark}")
print(f"#draw_individually: {draw_individually}")
print(f"#no_auto_figsize_adjust: {no_auto_figsize_adjust}")
print(f"#use_robust_zscore: {use_robust_zscore}")
print(f"#zscore_lowerbound: {zscore_lowerbound}")
print(f"#zscore_upperbound: {zscore_upperbound}")
print(f"#zscores_from_targets: {zscores_from_targets}")

### Functions
##
def parse_input (file, comment_escapes: list, field_sep: str, uncapitalize: bool = uncapitalize, remove_punctuations: bool = remove_punctuations, split_hyphenation: bool = split_hyphenation, check: bool = False) -> None:
    "reads a file, splits it into segments using a given separator, removes comments, and forward the result to main"
    ## reading data
    with file as f:
        lines =  [ line.strip() for line in f.readlines() if not line[0] in comment_escapes ]
    ##
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
    ## strip extra chars
    data = [ [ field.strip() for field in re.split(field_sep, line) ] for line in filtered_lines ]
    ## uncapitalize tokens
    if uncapitalize:
        data = [ [ x.lower() for x in l ] for l in data ]
    ## remove punctuations from lines
    punct_symbols = list(",.?!:;/\–~")
    if remove_punctuations:
        data = [ [ x for x in l if x not in punct_symbols ] for l in data ]
    ## split hyphenated tokens
    if split_hyphenation:
        data = [ process_hyphenation (l) for l in data ]
    ##
    return data

## process
if not file is None:
    S0 = parse_input (file, comment_escapes = input_comment_escapes, field_sep = input_field_sep, uncapitalize = uncapitalize, remove_punctuations = remove_punctuations, split_hyphenation = split_hyphenation, check = False)
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
Patterns = sorted (Patterns, key = lambda x: len(x), reverse = False)

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
    print(f"##Drawing diagrams individually")
    for i, patlat in enumerate(L):
        print(f"#drawing diagram from PatternLattice {i+1}")
        patlat.draw_diagrams (layout = layout, generalized = generalized, zscores_from_targets = zscores_from_targets, scale_factor = scale_factor, font_name = multibyte_font_name, check = draw_inspection)
    exit()

##
print(f"##Merging {len(L)} PatternLattices ...")
simplified     = False
label_sample_n = 5
if simplified:
    #print(f"#binary merger")
    La, Lb = L[0], L[1]
    if verbose:
        print(f"#La: {La}")
        print(f"#Lb: {Lb}")
    M = La.merge_lattices (Lb, use_mp = use_mp, show_steps = True, check = False)
elif build_lattice_stepwise:
    gen_links_internally = True
    print(f"##Mergig PatternLattices ...")
    for i, patlat in enumerate (L):
        print(f"#processing pattern lattice {i+1}")
        if i == 0:
            M = patlat
        else: ## merger
            M = M.merge_lattices (patlat, gen_links_internally = gen_links_internally, use_mp = use_mp, generalized = generalized, reflexive = reflexive, reductive = True, show_steps = True, check = False)
            ## delete the original
            patplat = None
        ## check nodes in M
        print(f"merged PatternLattice with {len(M.nodes)} nodes")
        for i, p in enumerate(M.nodes):
            print(f"#node {i:3d}: {p.separate_print()}")
        ## genenrate links in delay
        if len(M.links) == 0 and not gen_links_internally:
            ## Don't do: M = M.update(...)
            M.update_links (reflexive = reflexive, check = False) ## Crucially
        print(f"#generated {len(M.links)} links")
        ## checking links in M
        print(f"##Links")
        for i, link in enumerate(M.links):
            link.pprint (indicator = i, paired = True, link_type = "instantiates", check = False)
        ## generate z-scores from link targets
        gen_zscores_from_targets (M, gap_mark = gap_mark, use_robust_zscore = use_robust_zscore, check = False)
        if print_link_targets:
            for node, zscore in M.source_zscores.items():
                print(f"#node {node} has z-score {zscore: .3f}")
        ## generate z-scores from link sources
        gen_zscores_from_sources (M, gap_mark = gap_mark, use_robust_zscore = use_robust_zscore, check = False)
        for node, zscore in M.source_zscores.items():
            print(f"#node {node} has z-score {zscore: .3f}")
        ##
        print(f"##Results")
        M.draw_diagrams (layout = layout, generalized = generalized, label_sample_n = label_sample_n, use_robust_zscore = use_robust_zscore, zscore_lb = zscore_lowerbound, zscore_ub = zscore_upperbound, no_auto_figsize_adjust = no_auto_figsize_adjust,  font_name = multibyte_font_name, zscores_from_targets = zscores_from_targets, scale_factor = scale_factor, check = draw_inspection)
else:
    gen_links_internally = False
    M = functools.reduce (lambda La, Lb: La.merge_lattices (Lb, gen_links_internally = gen_links_internally, use_mp = use_mp, generalized = generalized, reflexive = reflexive, reductive = True, check = False), L)

    # The following process was isolated for memory conservation
    if len(M.links) == 0 and not gen_links_internally:
        print(f"##Generating links independently")
        ## Don't do: M = M.update(...)
        M.update_links (reflexive = reflexive, use_mp = use_mp, check = False)

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
    gen_zscores_from_targets (M, gap_mark = gap_mark, use_robust_zscore = use_robust_zscore, check = False)
    if print_link_targets:
        for node, zscore in M.source_zscores.items():
            print(f"#node {node} has z-score {zscore: .3f} [n: {M.link_targets[node]:2d}]")

    ## get z-scores from link sources
    gen_zscores_from_sources (M, gap_mark = gap_mark, use_robust_zscore = use_robust_zscore, check = False)
    for node, zscore in M.source_zscores.items():
        print(f"#node {node} has z-score {zscore: .3f} (n: {M.link_sources[node]:2d})")

    ## draw diagram of M
    print(f"##Drawing a diagram from the merged PatternLattice")
    M.draw_diagrams (layout = layout, generalized = generalized, label_sample_n = label_sample_n, use_robust_zscore = use_robust_zscore, zscore_lb = zscore_lowerbound, zscore_ub = zscore_upperbound, no_auto_figsize_adjust = no_auto_figsize_adjust, font_name = multibyte_font_name, zscores_from_targets = zscores_from_targets, scale_factor = scale_factor, check = draw_inspection)

## conclude
print(f"##built from {len(S)} sources: {[ as_label(x, sep = ',') for x in S ]}")

### end of file
