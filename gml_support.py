"""
GML Format Support for gPLB Pattern Lattices

GraphML (.gml) is a standard XML-based graph format supported by:
- NetworkX (Python)
- Gephi (visualization)
- Cytoscape (network analysis)
- igraph (R/Python)

Benefits over pickle:
- Human-readable XML
- Cross-platform compatible
- Can be opened in other graph tools
- Version-independent
- Easy to inspect and debug
"""

import networkx as nx
from pathlib import Path


def pattern_lattice_to_gml(lattice, output_file: str, 
                           include_zscores: bool = True,
                           include_content: bool = True,
                           check: bool = False):
    """
    Convert a PatternLattice to GML format and save.
    
    Args:
        lattice: PatternLattice object
        output_file: Output .gml filename
        include_zscores: Include z-score attributes
        include_content: Include pattern content (can make file large)
        check: Print debug info
    
    Example:
        pattern_lattice_to_gml(M, 'output.gml')
    """
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add metadata as graph attributes
    G.graph['generality'] = lattice.generality
    G.graph['p_metric'] = lattice.p_metric
    G.graph['gap_mark'] = lattice.gap_mark
    G.graph['tracer'] = lattice.tracer if hasattr(lattice, 'tracer') else '~'
    G.graph['n_nodes'] = len(lattice.nodes)
    G.graph['n_links'] = len(lattice.links)
    
    # Prepare z-scores dictionaries (convert tuple keys to strings)
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
    
    # Add nodes with attributes
    for pattern in lattice.nodes:
        # Node ID is string representation of form
        node_id = str(tuple(pattern.form))
        
        # Basic attributes
        node_attrs = {
            'label': ' '.join(pattern.form),  # Human-readable label
            'form': ','.join(pattern.form),   # Machine-readable form
            'rank': pattern.get_rank(),
            'gap_size': pattern.get_gap_size(),
            'size': len(pattern.form)
        }
        
        # Add content if requested
        if include_content and hasattr(pattern, 'content'):
            node_attrs['content'] = ','.join([','.join(c) if isinstance(c, (list, tuple)) else str(c) 
                                             for c in pattern.content])
        
        # Add z-scores if available
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
    
    # Add edges (links)
    for link in lattice.links:
        source_id = str(tuple(link.left.form))
        target_id = str(tuple(link.right.form))
        
        edge_attrs = {
            'link_type': link.link_type if link.link_type else 'instantiates',
            'link_rank': link.get_link_rank(),
            'link_gap_size': link.get_link_gap_size()
        }
        
        G.add_edge(source_id, target_id, **edge_attrs)
    
    # Save as GML
    nx.write_gml(G, output_file)
    
    if check:
        print(f"Saved lattice to: {output_file}")
        print(f"  Nodes: {G.number_of_nodes()}")
        print(f"  Edges: {G.number_of_edges()}")
        print(f"  Generality: G{lattice.generality}")
    
    return output_file


def load_gml_lattice(gml_file: str, check: bool = False):
    """
    Load a lattice from GML file.
    
    Args:
        gml_file: Input .gml filename
        check: Print debug info
    
    Returns:
        NetworkX DiGraph with all attributes
    
    Example:
        G = load_gml_lattice('output.gml')
        print(f"Nodes: {G.number_of_nodes()}")
        print(f"Generality: {G.graph['generality']}")
    """
    
    G = nx.read_gml(gml_file)
    
    if check:
        print(f"Loaded lattice from: {gml_file}")
        print(f"  Nodes: {G.number_of_nodes()}")
        print(f"  Edges: {G.number_of_edges()}")
        if 'generality' in G.graph:
            print(f"  Generality: G{G.graph['generality']}")
    
    return G


def gml_to_graphviz_dot(gml_file: str, output_file: str = None):
    """
    Convert GML to Graphviz DOT format for alternative visualization.
    
    Args:
        gml_file: Input .gml file
        output_file: Output .dot file (default: same name with .dot extension)
    
    Example:
        gml_to_graphviz_dot('lattice.gml', 'lattice.dot')
        # Then: dot -Tpng lattice.dot -o lattice.png
    """
    
    G = load_gml_lattice(gml_file)
    
    if output_file is None:
        output_file = Path(gml_file).with_suffix('.dot')
    
    # Convert to DOT
    from networkx.drawing.nx_pydot import write_dot
    write_dot(G, output_file)
    
    print(f"Converted to DOT format: {output_file}")
    print(f"  Visualize with: dot -Tpng {output_file} -o output.png")
    
    return output_file


def inspect_gml_metadata(gml_file: str):
    """
    Print metadata from GML file without loading full graph.
    
    Useful for quickly checking what's in a file.
    
    Example:
        inspect_gml_metadata('lattice.gml')
    """
    
    G = load_gml_lattice(gml_file)
    
    print(f"\n{'='*60}")
    print(f"GML File: {gml_file}")
    print(f"{'='*60}")
    
    # Graph metadata
    print("\nGraph Metadata:")
    for key, value in G.graph.items():
        print(f"  {key}: {value}")
    
    # Node statistics
    print(f"\nNode Statistics:")
    print(f"  Total nodes: {G.number_of_nodes()}")
    
    if G.number_of_nodes() > 0:
        # Sample node attributes
        sample_node = list(G.nodes(data=True))[0]
        print(f"\nNode Attributes (example):")
        for key, value in sample_node[1].items():
            print(f"  {key}: {value}")
        
        # Rank distribution
        ranks = [data.get('rank', 0) for _, data in G.nodes(data=True)]
        print(f"\nRank Distribution:")
        print(f"  Min: {min(ranks)}")
        print(f"  Max: {max(ranks)}")
        print(f"  Mean: {sum(ranks)/len(ranks):.2f}")
        
        # Z-score ranges (if available)
        source_zscores = [data.get('source_zscore') for _, data in G.nodes(data=True) 
                         if 'source_zscore' in data]
        if source_zscores:
            print(f"\nSource Z-scores:")
            print(f"  Min: {min(source_zscores):.3f}")
            print(f"  Max: {max(source_zscores):.3f}")
            print(f"  Mean: {sum(source_zscores)/len(source_zscores):.3f}")
    
    # Edge statistics
    print(f"\nEdge Statistics:")
    print(f"  Total edges: {G.number_of_edges()}")
    
    if G.number_of_edges() > 0:
        # Sample edge attributes
        sample_edge = list(G.edges(data=True))[0]
        print(f"\nEdge Attributes (example):")
        for key, value in sample_edge[2].items():
            print(f"  {key}: {value}")
    
    print(f"\n{'='*60}\n")


# ============================================================
# Integration with existing gPLB code
# ============================================================

def add_gml_export_to_main():
    """
    Instructions for adding GML export to __main__.py
    """
    
    instructions = """
    TO ADD GML EXPORT TO __main__.py:
    
    1. Add command-line argument (after line 106):
    
       parser.add_argument('--output_gml', '-g', type=str, default=None,
                          help='Save lattice as GML file (default: auto-generate from input name)')
       parser.add_argument('--no_auto_gml', action='store_true', default=False,
                          help='Disable automatic GML output')
    
    2. Parse arguments (after line 127):
    
       output_gml_file = args.output_gml
       auto_gml = not args.no_auto_gml
    
    3. After building merged lattice M (after line 543):
    
       # Auto-generate GML filename if not specified
       if auto_gml and output_gml_file is None:
           output_gml_file = f"g{generality}PL-{input_file_name_stem}.gml"
       
       # Save as GML
       if output_gml_file or auto_gml:
           print(f"##Saving lattice as GML")
           pattern_lattice_to_gml(M, output_gml_file, 
                                 include_zscores=True, 
                                 check=True)
    
    4. CHANGE DEFAULT BEHAVIOR: Skip drawing by default
    
       Replace line 72:
       OLD: parser.add_argument('-D', '--draw_instead_of_save', action='store_true', default=False)
       NEW: parser.add_argument('-D', '--draw', action='store_true', default=False,
                               help='Draw lattice (default: only save GML)')
       
       Then at line 542, wrap drawing in condition:
       
       if args.draw:  # Only draw if explicitly requested
           print(f"##Drawing a diagram from the merged PL")
           M.draw_lattice(...)
       else:
           print(f"##Skipped drawing (use -D/--draw to enable)")
    """
    
    print(instructions)


if __name__ == "__main__":
    print(__doc__)
    print("\n" + "="*60)
    print("GML FORMAT EXAMPLE")
    print("="*60)
    
    example_gml = """
    <?xml version="1.0" encoding="UTF-8"?>
    <graphml>
      <graph edgedefault="directed">
        <key id="generality" for="graph" attr.name="generality" attr.type="int"/>
        <key id="label" for="node" attr.name="label" attr.type="string"/>
        <key id="rank" for="node" attr.name="rank" attr.type="int"/>
        <key id="gap_size" for="node" attr.name="gap_size" attr.type="int"/>
        <key id="source_zscore" for="node" attr.name="source_zscore" attr.type="double"/>
        
        <data key="generality">3</data>
        
        <node id="('a', 'b', 'c')">
          <data key="label">a b c</data>
          <data key="rank">3</data>
          <data key="gap_size">0</data>
          <data key="source_zscore">2.45</data>
        </node>
        
        <node id="('_', 'a', 'b', 'c')">
          <data key="label">_ a b c</data>
          <data key="rank">3</data>
          <data key="gap_size">1</data>
          <data key="source_zscore">1.23</data>
        </node>
        
        <edge source="('_', 'a', 'b', 'c')" target="('a', 'b', 'c')">
          <data key="link_type">instantiates</data>
        </edge>
      </graph>
    </graphml>
    """
    
    print(example_gml)
    print("\n" + "="*60)
    print("USAGE EXAMPLES")
    print("="*60)
    print("""
    # Save lattice as GML (in gPLB code):
    pattern_lattice_to_gml(M, 'output.gml')
    
    # Load and inspect:
    G = load_gml_lattice('output.gml')
    inspect_gml_metadata('output.gml')
    
    # Convert to other formats:
    gml_to_graphviz_dot('output.gml', 'output.dot')
    """)
