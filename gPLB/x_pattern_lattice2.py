
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

##
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
def calc_averages_by (metric: str, link_dict: dict, grouped_links: dict, check: bool = False) -> dict:
    """
    calculate averages per a given metric, such as 'rank', 'gap_size'
    """

    assert metric in ['gap_size', 'rank']

    if check:
        print(f"#grouped_links: {grouped_links}")
    ##
    averages_by = {}
    for metric_val in grouped_links:
        members = grouped_links[metric_val]
        dist = [ link_dict[m] for m in members ]
        averages_by[metric_val] = sum(dist)/len(dist)
    ##
    return averages_by

##
def calc_stdevs_by (metric: str, link_dict: dict, grouped_links: dict, check: bool = False) -> dict:
    """
    calculate stdevs per a given rank such as 'rank', 'gap_size'
    """

    assert metric in ['rank', 'gap_size']

    if check:
        print(f"#grouped_links: {grouped_links}")
    ##
    import numpy as np
    stdevs_by = {}
    for metric_val in grouped_links:
        members = grouped_links[metric_val]
        dist = [ link_dict[m] for m in members ]
        stdevs_by[metric_val] = np.std(dist)
    ##
    return stdevs_by

##
def calc_medians_by (metric: str, link_dict: dict, grouped_links: dict, check: bool = False) -> dict:
    """
    calculate stdevs per a given metric such as 'rank', 'gap_size'
    """

    assert metric in ['rank', 'gap_size']
    if check:
        print(f"#grouped_links: {grouped_links}")
    ##
    import numpy as np
    medians_by = {}
    for metric_val in grouped_links:
        members = grouped_links[metric_val]
        dist = [ link_dict[m] for m in members ]
        medians_by[metric_val] = np.median(dist)
    ##
    return medians_by


##
def calc_MADs_by (metric: str, link_dict: dict, grouped_links: dict, check: bool = False) -> dict:
    """
    calculate stdevs per a given metric such as 'rank', 'gap_size'
    """

    assert metric in ['rank', 'gap_size']
    if check:
        print(f"#ranked_links: {ranked_links}")

    ## JIT compiler demand function-internal imports to be externalized
    import numpy as np
    import scipy.stats as stats
    ##
    MADs_by = {}
    for metric_val in grouped_links:
        members = grouped_links[metric_val]
        dist = [ link_dict[m] for m in members ]
        MADs_by[metric_val] = np.median (stats.median_abs_deviation (dist))
    ##
    return MADs_by

##
def calc_MAD (dist):
    """Calculate Median Absolute Deviation (MAD) with proper scaling."""
    import numpy as np
    import scipy.stats as stats
    return np.median(stats.median_abs_deviation (dist))

# In pattern_lattice.py, replace lines 406-559 with:
