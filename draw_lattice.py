#!/usr/bin/env python3
"""
draw_lattice.py - Draw gPLB lattices from GML files

This script reads a .gml file created by gPLB and draws the lattice
with visualization that matches the original draw_graph() behavior.

Usage:
    python draw_lattice.py lattice.gml
    python draw_lattice.py lattice.gml --layout spring
    python draw_lattice.py lattice.gml --zscore_lb -1 --zscore_ub 2
    python draw_lattice.py lattice.gml --fig_size 15,15 --show

Features:
    - Multiple layout algorithms (Multi_partite, spring, kamada_kawai, etc.)
    - Z-score filtering
    - Customizable figure size and DPI
    - Interactive display or save to file
    - Auto-sizing based on graph complexity
    - Matches original gPLB draw_graph() styling
"""

import argparse
import sys
from pathlib import Path
import networkx as nx
import math

##
def load_gml_lattice(gml_file: str, verbose: bool = False):
    """Load lattice from GML file"""
    try:
        G = nx.read_gml(gml_file)

        if verbose:
            print(f"✓ Loaded lattice from: {gml_file}")
            print(f"  Nodes: {G.number_of_nodes()}")
            print(f"  Edges: {G.number_of_edges()}")
            if 'generality' in G.graph:
                print(f"  Generality: G{G.graph['generality']}")

        return G

    except Exception as e:
        print(f"Error loading GML file: {e}")
        sys.exit(1)

##
def filter_graph_by_zscore(G, zscore_attr: str = None, lb: float = None, ub: float = None, verbose: bool = False):
    """Filter graph nodes by z-score range"""

    # Auto-detect z-score attribute if not specified
    if zscore_attr is None:
        zscore_attr = auto_detect_zscore_attr(G, verbose=verbose)
        if zscore_attr is None:
            print(f"Warning: No z-score attribute found for filtering. Returning unfiltered graph.")
            return G

    # Check if z-score attribute exists
    node_attrs = next(iter(G.nodes(data=True)), (None, {}))[1]
    if zscore_attr not in node_attrs:
        print(f"Warning: Z-score attribute '{zscore_attr}' not found in nodes")
        print(f"Available attributes: {list(node_attrs.keys())}")
        # Try auto-detection as fallback
        alt_attr = auto_detect_zscore_attr(G, verbose=verbose)
        if alt_attr and alt_attr != zscore_attr:
            print(f"Using auto-detected attribute '{alt_attr}' instead.")
            zscore_attr = alt_attr
        else:
            return G

    # Filter nodes
    original_size = G.number_of_nodes()

    filtered_nodes = []
    for node, data in G.nodes(data=True):
        zscore = data.get(zscore_attr)
        if zscore is None:
            continue

        # Apply filters
        if lb is not None and zscore < lb:
            continue
        if ub is not None and zscore > ub:
            continue

        filtered_nodes.append(node)

    # Create subgraph
    G_filtered = G.subgraph(filtered_nodes).copy()

    if verbose:
        removed = original_size - G_filtered.number_of_nodes()
        print(f"Z-score filtering ({zscore_attr}): {lb} to {ub}")
        print(f"  Kept: {G_filtered.number_of_nodes()} nodes")
        print(f"  Removed: {removed} nodes")

    return G_filtered

##
def auto_detect_zscore_attr(G, preferred_attr: str = None, verbose: bool = False):
    """
    Auto-detect which z-score attribute to use.
    Priority: preferred_attr > source_robust_zscore > source_zscore > target_robust_zscore > target_zscore
    """
    # Get available attributes from first node
    if G.number_of_nodes() == 0:
        return None

    sample_node_data = list(G.nodes(data=True))[0][1]
    available_attrs = list(sample_node_data.keys())

    # Define priority order for z-score attributes
    zscore_attr_priority = [
        'source_robust_zscore',  # Most commonly used
        'source_zscore',
        'target_robust_zscore',
        'target_zscore'
    ]

    # If preferred attribute specified, try it first
    if preferred_attr:
        if preferred_attr in available_attrs:
            if verbose:
                print(f"# Using specified z-score attribute: '{preferred_attr}'")
            return preferred_attr
        else:
            print(f"Warning: Specified z-score attribute '{preferred_attr}' not found.")
            print(f"Available attributes: {available_attrs}")
            print(f"Trying to auto-detect alternative...")

    # Try attributes in priority order
    for attr in zscore_attr_priority:
        if attr in available_attrs:
            if verbose:
                print(f"# Auto-detected z-score attribute: '{attr}'")
            return attr

    # No z-score attribute found
    if verbose:
        print(f"# No z-score attribute found in: {available_attrs}")
    return None

##
def draw_lattice_from_gml(G,
                         input_name: str,
                         layout: str = 'multipartite',
                         multipartite_key: str = 'gap_size',
                         fig_size: tuple = None,
                         fig_dpi: int = 360,
                         node_size: int = None,
                         font_size: int = None,
                         output_file: str = None,
                         show: bool = False,
                         zscore_attr: str = None,  # Changed to None for auto-detection
                         color_by_zscore: bool = True,
                         use_curved_edges: bool = True,
                         label_offset: tuple = None,
                         scale_factor: float = 3,
                         highlight_links_around: list = [],
                         label_sample_n: int = None,
                         graphics_backend: str = "qt",
                         verbose: bool = False):
    """Draw lattice using matplotlib/networkx matching original gPLB draw_graph() behavior"""

    import matplotlib.pyplot as plt
    from matplotlib import colormaps

    # set backend
    import matplotlib
    if graphics_backend is None:
        matplotlib.use('Agg')
    elif graphics_backend == 'qt':
        matplotlib.use('Qt5Agg') # effective
    elif graphics_backend == 'tk':
        matplotlib.use('TkAgg') # default and not effective
    elif graphics_backend == 'gtk':
        matplotlib.use('GTK3Agg') # requires install
    elif graphics_backend == 'wx':
        matplotlib.use('WXAgg') # requires install and not effective
    else:
        matplotlib.use('Qt5Agg')

    # Store original node count for filtering stats
    n_original_nodes = G.number_of_nodes()
    n_nodes = n_original_nodes

    # Auto-detect z-score attribute if not specified
    if color_by_zscore:
        detected_zscore_attr = auto_detect_zscore_attr(G, preferred_attr=zscore_attr, verbose=verbose)
        if detected_zscore_attr:
            zscore_attr = detected_zscore_attr
        else:
            print("Warning: No z-score attribute found. Disabling z-score coloring.")
            color_by_zscore = False

    # Extract instances (patterns with gap_size == 0) for title
    instances = []
    for node, data in G.nodes(data=True):
        gap_size = data.get('gap_size', None)
        if gap_size == 0:
            literal = data.get('literal')
            if literal and literal not in instances:
                instances.append(literal)

    if verbose:
        print(f"# Found {len(instances)} instances: {instances[:5]}...")  # Show first 5

    # Calculate layout
    if verbose:
        print(f"Calculating layout: {layout}")

    # Set node positions based on layout
    if layout in ['multipartite', 'multi_partite', 'Multipartite', 'Multi-partite', 'Multi_partite', 'M', 'MP', 'mp']:
        # Check if the attribute exists in nodes
        node_attrs = nx.get_node_attributes(G, multipartite_key)

        if verbose:
            print(f"Checking for attribute '{multipartite_key}'")
            print(f"Found {len(node_attrs)} nodes with this attribute")

        if len(node_attrs) == 0:
            print(f"Warning: Attribute '{multipartite_key}' not found in nodes. Using spring layout.")
            if G.nodes():
                sample_node = list(G.nodes(data=True))[0]
                available_attrs = list(sample_node[1].keys())
                print(f"Available node attributes: {available_attrs}")
                if 'rank' in available_attrs:
                    print(f"Hint: Try --mp_key rank")
            layout_name = "Spring"
            pos = nx.spring_layout(G, k=1.4, dim=2)
            connectionstyle = "arc"
        else:
            layout_name = "Multi-partite"
            # Use scale=-1 like original
            pos = nx.multipartite_layout(G, subset_key=multipartite_key, scale=-1)

            # Flip x-coordinates when rank is used (like original)
            if multipartite_key in ['rank']:
                pos = {node: (-x, y) for node, (x, y) in pos.items()}

            # Use curved edges for multipartite (like original gPLB)
            connectionstyle = "arc, angleA=0, angleB=180, armA=50, armB=50, rad=15" if use_curved_edges else "arc"

    elif layout in ['spring', 'Spring', 'Sp']:
        layout_name = "Spring"
        pos = nx.spring_layout(G, k=1.4, dim=2)
        connectionstyle = "arc"

    elif layout in ['kamada_kawai', 'Kamada-Kawai', 'Kamada_Kawai', 'KK']:
        layout_name = "Kamada-Kawai"
        pos = nx.kamada_kawai_layout(G, scale=scale_factor, dim=2)
        connectionstyle = "arc"

    elif layout in ['circular', 'Circular', 'C']:
        layout_name = "Circular"
        pos = nx.circular_layout(G, scale=scale_factor, dim=2)
        connectionstyle = "arc"

    elif layout in ['shell', 'Shell', 'Sh']:
        layout_name = "Shell"
        pos = nx.shell_layout(G, scale=scale_factor, dim=2)
        connectionstyle = "arc"

    elif layout in ['spectral', 'Spectral', 'Spc']:
        layout_name = "Spectral"
        pos = nx.spectral_layout(G, scale=scale_factor, dim=2)
        connectionstyle = "arc"

    else:
        print(f"Unknown layout: {layout}. Using spring.")
        layout_name = "Spring"
        pos = nx.spring_layout(G, k=1.4, dim=2)
        connectionstyle = "arc"

    # Get MPG key stats for auto-sizing
    MPG_keys = nx.get_node_attributes(G, multipartite_key)
    MPG_key_count_max = 1
    MPG_group_size = 1
    if MPG_keys:
        import collections
        MPG_key_counts = collections.defaultdict(int)
        for k, v in MPG_keys.items():
            MPG_key_counts[v] += 1
        if verbose:
            print(f"#MPG_key_counts: {MPG_key_counts}")
        MPG_key_count_max = max(MPG_key_counts.values())
        print(f"# MPG_key_count_max: {MPG_key_count_max}")
        MPG_group_size = len(MPG_key_counts.keys())
        print(f"# MPG_group_size: {MPG_group_size}")

    # Auto-size figure if requested (matching original logic)
    if fig_size is None:
        n = G.number_of_nodes()
        m = G.number_of_edges()
        base_size = 4
        width_scale_factor = 0.3
        height_scale_factor = 0.3

        graph_width = round(base_size + width_scale_factor * ((m + n) ** 0.5), 0)
        graph_height = round(base_size + height_scale_factor * ((m + n) ** 0.6), 0)

        if layout_name in ["Multi-partite", "Multi_partite", "Multipartite", "MP"]:
            fig_size = (graph_width, graph_height)
        else:
            fig_size = (graph_width, graph_width)

    print(f"# fig_size: {fig_size}")
    print(f"# fig_dpi: {fig_dpi}")

    # Create figure (don't specify dpi here to avoid connectionstyle issues)
    plt.figure(figsize=fig_size)

    # Auto-size node and label sizes (matching original)
    resize_coeff = 0.33 # 0.67 formerly
    if fig_size is None:
        if node_size is None:
            node_size = 10 - round(resize_coeff * math.log(1 + MPG_key_count_max), 0)
        if font_size is None:
            font_size = 8 - round(resize_coeff * math.log(1 + MPG_key_count_max), 0)
    else:
        if node_size is None:
            node_size = 10
        if font_size is None:
            font_size = 8

    print(f"# node_size: {node_size}")
    print(f"# label_size: {font_size}")

    # Determine node colors based on z-scores (matching original)
    if color_by_zscore:
        node_colors = []
        zscore_values = []
        for node in G.nodes():
            zscore = G.nodes[node].get(zscore_attr, 0.0)
            node_colors.append(zscore)
            zscore_values.append(zscore)

        # Print z-score statistics for debugging
        if zscore_values:
            min_z = min(zscore_values)
            max_z = max(zscore_values)
            avg_z = sum(zscore_values) / len(zscore_values)
            print(f"# z-score range: [{min_z:.3f}, {max_z:.3f}], avg: {avg_z:.3f}")
            if verbose:
                non_zero = [z for z in zscore_values if z != 0.0]
                print(f"# Non-zero z-scores: {len(non_zero)} out of {len(zscore_values)}")
                if non_zero:
                    print(f"# Sample non-zero z-scores: {non_zero[:5]}")
    else:
        node_colors = [0.0] * len(G.nodes())

    # Draw nodes with coolwarm colormap (like original)
    my_cmap = colormaps['coolwarm']
    nx.draw_networkx_nodes(G, pos,
                          node_size=node_size,
                          node_color=node_colors,
                          cmap=my_cmap)

    # Differentiate edge colors
    if len(highlight_links_around):
        print(f"# highlight_links_around: {highlight_links_around}")
    edge_colors = []
    edge_widths = []
    for i, edge in enumerate(G.edges()):
        e0, e1 = eval(edge[0]), eval(edge[1]) # Crucially, eval(..)
        # Both nodes selected
        if e0 in highlight_links_around and e1 in highlight_links_around:
            print(f"found two hightlight targets: {e0} and {e1}")
            edge_colors.append('red')
            edge_widths.append(0.2)
        # One node selected
        elif e0 in highlight_links_around or e1 in highlight_links_around:
            print(f"found one hightlight target: {e0} or {e1}")
            edge_colors.append('orange')
            edge_widths.append(0.2)
        # No nodes selected
        else:
            edge_colors.append('gray')
            edge_widths.append(0.05)

    # Draw edges matching original styling
    nx.draw_networkx_edges(G, pos,
                          edge_color=edge_colors, #edge_color='gray',
                          width=edge_widths, #width=0.05,  # Original uses 0.05
                          arrowsize=5,  # Original uses 5
                          arrows=True,
                          connectionstyle=connectionstyle,
                          min_source_margin=0.1,
                          min_target_margin=0.1)

    # Create label positions with offset (matching original defaults)
    if label_offset is None:
        label_offset = (0, 0)  # Original defaults

    label_offset_x, label_offset_y = label_offset
    label_positions = {
        node: (x + label_offset_x, y + label_offset_y)
        for node, (x, y) in pos.items()
    }

    # Draw labels - use literal attribute like original, fall back to form
    labels = {}
    for node in G.nodes():
        # Try literal attribute first (like original)
        literal = G.nodes[node].get('literal')
        if literal:
            # Convert tuple/list to space-separated string
            if isinstance(literal, (list, tuple)):
                label = ' '.join(str(x) for x in literal)
            else:
                label = str(literal)
        else:
            # Fall back to form attribute
            form = G.nodes[node].get('form')
            if form:
                if isinstance(form, str):
                    label = form.replace(',', ' ')
                else:
                    label = ' '.join(str(x) for x in form).replace(',', ' ')
            else:
                # Last resort: use node ID
                label = str(node).strip("()").replace("'", "").replace(", ", " ")
        labels[node] = label

    nx.draw_networkx_labels(G, label_positions,
                           labels=labels,
                           font_size=font_size,
                           font_color='darkblue',
                           verticalalignment='top',
                           horizontalalignment='left')

    # Prepare instance labels for title (matching original)
    instance_labels = []
    for instance in instances:
        if isinstance(instance, (list, tuple)):
            # Join with comma like original as_label(x, sep=",")
            label = ','.join(str(x) for x in instance)
        else:
            label = str(instance)
        instance_labels.append(label)

    label_count = len(instance_labels)

    # Sample labels if too many (matching original behavior)
    if label_sample_n is not None and label_count > label_sample_n:
        new_instance_labels = instance_labels[:label_sample_n - 1]
        new_instance_labels.append("…")
        new_instance_labels.append(instance_labels[-1])
        instance_labels = new_instance_labels

    if verbose:
        print(f"#instance_labels {label_count}: {instance_labels}")

    # Title (matching original format more closely)
    title_parts = []
    if 'generality' in G.graph:
        pl_type = f"g{G.graph['generality']}"
    else:
        pl_type = "gX"

    if layout_name in ['Multi-partite']:
        layout_desc = f"{layout_name} [key: {multipartite_key}]"
    else:
        layout_desc = layout_name

    # Build title similar to original
    title_val = f"{pl_type} (layout: {layout_desc})"

    # Add instance information to title
    if instance_labels:
        title_val += f"\nbuilt from {instance_labels} ({label_count} in all)"

    plt.title(title_val, fontsize=10)
    plt.axis('off')
    plt.tight_layout()

    # Save or show
    if show:
        if verbose:
            print("Displaying figure...")
        plt.show()
    else:
        if output_file is None:
            if input_name:
                output_file = f"{input_name}.png"
            else:
                #output_file = 'lattice_output.png'
                output_file = f"g{G.graph['generality']}PL.png"
        # Don't specify dpi in savefig to avoid connectionstyle issues (like original)
        plt.savefig(output_file)
        print(f"✓ Saved figure to: {output_file}")

    plt.close()

##
def main():
    parser = argparse.ArgumentParser(
        description='Draw gPLB lattices from GML files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python draw_lattice.py lattice.gml

  # Different layouts
  python draw_lattice.py lattice.gml --layout spring
  python draw_lattice.py lattice.gml --layout kamada_kawai
  python draw_lattice.py lattice.gml --layout circular

  # Filter by z-score
  python draw_lattice.py lattice.gml --zscore_lb -1 --zscore_ub 2

  # Customize appearance
  python draw_lattice.py lattice.gml --fig_size 15,15 --node_size 500 --font_size 10

  # High resolution output
  python draw_lattice.py lattice.gml --fig_dpi 600 -o high_res_output.png

  # Interactive display
  python draw_lattice.py lattice.gml --show

  # Multi-partite by rank instead of gap_size
  python draw_lattice.py lattice.gml --layout multipartite --mp_key rank

  # Disable auto-sizing
  python draw_lattice.py lattice.gml --no_auto_sizing --fig_size 12,10
        """
    )

    # Required arguments
    parser.add_argument('gml_file', type=str,
                       help='Input GML file from gPLB')

    # Layout options
    parser.add_argument('--layout', '-L', type=str, default='multipartite',
                       choices=['multipartite', 'multi_partite', 'spring', 'kamada_kawai',
                               'circular', 'shell', 'spectral'],
                       help='Layout algorithm (default: multipartite)')
    parser.add_argument('--mp_key', type=str, default='gap_size',
                       choices=['gap_size', 'rank'],
                       help='Multipartite grouping key (default: gap_size)')

    # Figure options
    parser.add_argument('--fig_size', '-F', type=str, default=None,
                       help='Figure size as width,height (e.g., 10,9). Auto-sized if not specified.')
    parser.add_argument('--fig_dpi', type=int, default=360,
                       help='Figure DPI (default: 360)')
    parser.add_argument('--node_size', type=int, default=None,
                       help='Node size. Auto-sized if not specified.')
    parser.add_argument('--font_size', type=int, default=None,
                       help='Font size. Auto-sized if not specified.')
    parser.add_argument('--no_auto_sizing', action='store_true',
                       help='Disable automatic figure and node sizing')
    parser.add_argument('--scale_factor', type=float, default=3,
                       help='Scale factor for some layouts (default: 3)')
    parser.add_argument('--highlight_links_around', '-H', type=str, default=None,
                       help='Differentiate links connected specified nodes by color')

    # Z-score filtering
    parser.add_argument('--zscore_lb', '-l', type=float, default=None,
                       help='Z-score lower bound')
    parser.add_argument('--zscore_ub', '-u', type=float, default=None,
                       help='Z-score upper bound')
    parser.add_argument('--zscore_attr', '-z', type=str, default='source_robust_zscore',
                       help='Z-score attribute to use (default: auto-detect). '
                            'Options: source_robust_zscore, source_zscore, target_robust_zscore, target_zscore')
    parser.add_argument('--no_color', '-C', action='store_true',
                       help='Disable z-score coloring')

    # Output options
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='Output filename (auto-generated if not specified)')
    parser.add_argument('--show', '-D', action='store_true',
                       help='Display interactively instead of saving')

    # Additional options
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output')
    parser.add_argument('--inspect', action='store_true',
                       help='Inspect GML metadata and exit')
    parser.add_argument('--no_curved_edges', action='store_true',
                       help='Disable curved edges for multipartite layout')
    parser.add_argument('--label_offset', type=str, default=None,
                       help='Label offset as x,y (default: 0.003,0.003)')
    parser.add_argument('--label_sample_n', type=int, default=None,
                       help='Maximum number of instance labels to show in title (default: show all)')

    args = parser.parse_args()

    # Check if file exists
    file = Path(args.gml_file)
    if not file.exists():
        print(f"Error: file not found: {args.gml_file}")
        sys.exit(1)

    # input_name
    #input_file_name = file.name
    input_name = Path(file.name).stem

    # Load GML
    G = load_gml_lattice(args.gml_file, verbose=args.verbose)

    # Inspect mode
    if args.inspect:
        print(f"\n{'='*60}")
        print(f"GML file: {args.gml_file}")
        print(f"{'='*60}")
        print(f"\nGraph metadata:")
        for key, value in G.graph.items():
            print(f"  {key}: {value}")

        print(f"\nNode statistics:")
        print(f"  Total: {G.number_of_nodes()}")
        if G.number_of_nodes() > 0:
            sample_node = list(G.nodes(data=True))[0]
            print(f"  Attributes: {list(sample_node[1].keys())}")

        print(f"\nEdge statistics:")
        print(f"  Total: {G.number_of_edges()}")
        print(f"{'='*60}\n")
        sys.exit(0)

    # Parse fig_size
    if args.fig_size:
        try:
            fig_size = tuple(map(float, args.fig_size.split(',')))
            if len(fig_size) != 2:
                raise ValueError
        except ValueError:
            print(f"Error: Invalid fig_size format. Use: width,height (e.g., 10,9)")
            sys.exit(1)
    else:
        fig_size = None

    # Parse label_offset
    if args.label_offset:
        try:
            label_offset = tuple(map(float, args.label_offset.split(',')))
            if len(label_offset) != 2:
                raise ValueError
        except ValueError:
            print(f"Error: Invalid label_offset format. Use: x,y (e.g., 0.01,0.01)")
            sys.exit(1)
    else:
        label_offset = None

    # Filter by z-score if requested
    if args.zscore_lb is not None or args.zscore_ub is not None:
        G = filter_graph_by_zscore(G, zscore_attr=args.zscore_attr,
                                   lb=args.zscore_lb,
                                   ub=args.zscore_ub,
                                   verbose=args.verbose)

    # Draw
    if args.verbose:
        print(f"\nDrawing lattice...")
        print(f"  Layout: {args.layout}")
        if args.layout in ['multipartite', 'multi_partite']:
            print(f"  Multipartite key: {args.mp_key}")
        if fig_size:
            print(f"  Figure size: {fig_size}")
        else:
            print(f"  Figure size: auto")
        print(f"  Output: {'[display]' if args.show else args.output or '[auto]'}")

    ##
    import gPLB.pattern as plb
    highlight_targets = [ tuple(x.split(",")) for x in args.highlight_links_around.split(";") ]
    print(f"# highlight_targets: {highlight_targets}")
    ##
    draw_lattice_from_gml(
        G,
        input_name,
        layout=args.layout,
        multipartite_key=args.mp_key,
        fig_size=fig_size,
        fig_dpi=args.fig_dpi,
        node_size=args.node_size,
        font_size=args.font_size,
        output_file=args.output,
        show=args.show,
        zscore_attr=args.zscore_attr,
        color_by_zscore=not args.no_color,
        use_curved_edges=not args.no_curved_edges,
        label_offset=label_offset,
        scale_factor=args.scale_factor,
        highlight_links_around=highlight_targets,
        label_sample_n=args.label_sample_n,
        verbose=args.verbose
    )

    if args.verbose:
        print("\n✓ Complete!")

##
if __name__ == "__main__":
    main()
