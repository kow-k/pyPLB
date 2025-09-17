## checking link target z-scores to M
if verbose:
    print(f"##Link_targets")
Links = M.link_
averages_by_rank = calc_averages_by_rank (M.link_targets) # returns dictionary
stdevs_by_rank   = calc_stdevs_by_rank (M.link_targets) # returns dictionary
stdevs_by_rank   = calc_medians_by_rank (M.link_targets) # returns dictionary

target_zscores = {}
for i, link_target in enumerate(M.link_targets):
    rank   = get_rank_of_list (link_target)
    value  = M.link_targets[link_target]
    #zscore = calc_zscore_old (value, averages_by_rank[rank], stdevs_by_rank[rank])
    zscore = calc_zscore (value, averages_by_rank[rank], stdevs_by_rank[rank], medians_by_rank[rank], MADs_by_rank[rank])
    target_zscores[link_target] = zscore
    if verbose:
        print(f"#link_target {i:3d}: {link_target} has {value} parent(s) [{target_zscores[link_target]:.6f} at rank {rank}]")
## attach target_zscores to M
#M.target_zscores = target_zscores
M.target_zscores.update(target_zscores)
if verbose:
    print(f"M.target_zscores: {M.target_zscores}")
