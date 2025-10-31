#!/usr/bin/env python3
"""
inspect_gml.py - Diagnostic tool to inspect GML file contents

Usage:
    python inspect_gml.py your_file.gml
"""

import sys
import networkx as nx
from pathlib import Path

def inspect_gml(gml_file):
    """Inspect GML file and report its contents"""
    
    if not Path(gml_file).exists():
        print(f"Error: File not found: {gml_file}")
        return
    
    print(f"\n{'='*70}")
    print(f"Inspecting GML file: {gml_file}")
    print(f"{'='*70}")
    
    try:
        G = nx.read_gml(gml_file)
    except Exception as e:
        print(f"Error reading GML: {e}")
        return
    
    # Graph-level attributes
    print(f"\n📊 GRAPH METADATA:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"\n  Graph attributes:")
    for key, value in G.graph.items():
        print(f"    {key}: {value}")
    
    # Node attributes
    if G.number_of_nodes() > 0:
        print(f"\n🔵 NODE ATTRIBUTES:")
        
        # Get first node to see what attributes exist
        sample_node_id, sample_node_data = list(G.nodes(data=True))[0]
        print(f"\n  Sample node: {sample_node_id}")
        print(f"  Attributes found:")
        for key, value in sample_node_data.items():
            print(f"    - {key}: {value}")
        
        # Check for required attributes
        print(f"\n  Checking required attributes:")
        required = ['label', 'rank', 'gap_size', 'form']
        for attr in required:
            node_attrs = nx.get_node_attributes(G, attr)
            if node_attrs:
                print(f"    ✓ {attr}: Found in {len(node_attrs)} nodes")
                # Show sample values
                sample_values = list(node_attrs.values())[:3]
                print(f"      Sample values: {sample_values}")
            else:
                print(f"    ✗ {attr}: NOT FOUND")
        
        # Check for optional attributes (z-scores)
        print(f"\n  Checking optional attributes (z-scores):")
        optional = ['source_zscore', 'target_zscore', 'source_robust_zscore', 'target_robust_zscore']
        for attr in optional:
            node_attrs = nx.get_node_attributes(G, attr)
            if node_attrs:
                print(f"    ✓ {attr}: Found in {len(node_attrs)} nodes")
            else:
                print(f"    - {attr}: Not present")
        
        # Rank distribution
        ranks = nx.get_node_attributes(G, 'rank')
        if ranks:
            rank_values = list(ranks.values())
            print(f"\n  Rank distribution:")
            print(f"    Min: {min(rank_values)}")
            print(f"    Max: {max(rank_values)}")
            
            # Count by rank
            from collections import Counter
            rank_counts = Counter(rank_values)
            print(f"    Distribution:")
            for rank in sorted(rank_counts.keys()):
                print(f"      Rank {rank}: {rank_counts[rank]} nodes")
        
        # Gap size distribution
        gap_sizes = nx.get_node_attributes(G, 'gap_size')
        if gap_sizes:
            gap_values = list(gap_sizes.values())
            print(f"\n  Gap size distribution:")
            print(f"    Min: {min(gap_values)}")
            print(f"    Max: {max(gap_values)}")
            
            # Count by gap_size
            from collections import Counter
            gap_counts = Counter(gap_values)
            print(f"    Distribution:")
            for gap_size in sorted(gap_counts.keys()):
                print(f"      Gap size {gap_size}: {gap_counts[gap_size]} nodes")
    
    # Edge attributes
    if G.number_of_edges() > 0:
        print(f"\n🔗 EDGE ATTRIBUTES:")
        
        # Get first edge
        sample_edge = list(G.edges(data=True))[0]
        print(f"\n  Sample edge: {sample_edge[0]} → {sample_edge[1]}")
        print(f"  Attributes found:")
        for key, value in sample_edge[2].items():
            print(f"    - {key}: {value}")
    
    print(f"\n{'='*70}")
    print(f"✓ Inspection complete")
    print(f"{'='*70}\n")
    
    # Recommendations
    print(f"💡 RECOMMENDATIONS:")
    
    if G.number_of_nodes() == 0:
        print(f"  ⚠️  File has no nodes! Check if lattice was built correctly.")
    
    node_attrs = nx.get_node_attributes(G, 'gap_size')
    if not node_attrs and G.number_of_nodes() > 0:
        print(f"  ⚠️  'gap_size' attribute missing!")
        print(f"     This is needed for multipartite layout.")
        print(f"     Check if pattern_lattice_to_gml() is saving attributes correctly.")
    
    rank_attrs = nx.get_node_attributes(G, 'rank')
    if not rank_attrs and G.number_of_nodes() > 0:
        print(f"  ⚠️  'rank' attribute missing!")
        print(f"     This is needed for multipartite layout with --mp_key rank.")
    
    if node_attrs and rank_attrs:
        print(f"  ✓ File looks good! All required attributes present.")
        print(f"    You can use:")
        print(f"      python draw_lattice.py {gml_file} --layout multipartite")
        print(f"      python draw_lattice.py {gml_file} --layout multipartite --mp_key rank")
    
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_gml.py <gml_file>")
        print("\nExample:")
        print("  python inspect_gml.py g1PL-pl-drink.gml")
        sys.exit(1)
    
    inspect_gml(sys.argv[1])
