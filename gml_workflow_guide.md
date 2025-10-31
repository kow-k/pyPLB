# gPLB GML-Based Workflow Guide

## Overview

**NEW DEFAULT BEHAVIOR**: gPLB now builds lattices and saves them as `.gml` files **without drawing**. Drawing is a separate step using the `draw_lattice.py` script.

### Why This Change?

1. **Faster**: Build once, visualize many times
2. **Standard format**: GML works with Gephi, Cytoscape, igraph
3. **Flexible**: Change visualization without rebuilding
4. **Efficient**: Batch processing without drawing overhead
5. **Portable**: Share GML files, others can visualize

---

## Quick Start

### Step 1: Build Lattice (Saves GML)

```bash
python -m gPLB input.txt -G3
```

Output: `g3PL-input.gml`

### Step 2: Draw Lattice

```bash
python draw_lattice.py g3PL-input.gml
```

Output: `g3PL-input_viz.png`

**That's it!** Build once, draw many times with different settings.

---

## Installation

### Apply Code Patches

Follow instructions in `GML_PATCHES.py`:

1. Add `import networkx as nx` to `__main__.py`
2. Add `pattern_lattice_to_gml()` function
3. Change `-D` flag behavior (draw becomes opt-in)
4. Add GML saving after lattice building
5. Make drawing conditional on `-D` flag

### Dependencies

```bash
pip install networkx matplotlib
```

---

## Command Reference

### Building Lattices (gPLB)

**Default behavior** (build + save GML, no drawing):
```bash
python -m gPLB input.txt -G3
```

**Build and draw** (if you want both):
```bash
python -m gPLB input.txt -G3 -D
```

**Custom GML filename**:
```bash
python -m gPLB input.txt -G3 -g my_lattice.gml
```

**Disable GML output** (compatibility mode):
```bash
python -m gPLB input.txt -G3 -D --no_gml
```

### Drawing Lattices (draw_lattice.py)

**Basic usage**:
```bash
python draw_lattice.py lattice.gml
```

**Different layouts**:
```bash
python draw_lattice.py lattice.gml --layout spring
python draw_lattice.py lattice.gml --layout kamada_kawai
python draw_lattice.py lattice.gml --layout circular
```

**Filter by z-score**:
```bash
python draw_lattice.py lattice.gml --zscore_lb -1 --zscore_ub 2
```

**Customize appearance**:
```bash
python draw_lattice.py lattice.gml --fig_size 15,15 --node_size 500 --font_size 10
```

**High resolution**:
```bash
python draw_lattice.py lattice.gml --fig_dpi 600 -o high_res.png
```

**Interactive display**:
```bash
python draw_lattice.py lattice.gml --show
```

**Inspect GML file**:
```bash
python draw_lattice.py lattice.gml --inspect
```

---

## Usage Examples

### Example 1: Simple Workflow

```bash
# Build lattice (30 minutes)
python -m gPLB data.txt -G3

# Try different visualizations (10 seconds each)
python draw_lattice.py g3PL-data.gml --layout spring
python draw_lattice.py g3PL-data.gml --layout kamada_kawai
python draw_lattice.py g3PL-data.gml --layout multipartite --mp_key rank

# Time saved: 89 minutes!
```

### Example 2: Batch Processing

```bash
# Phase 1: Build all lattices (no drawing overhead!)
for file in data/*.txt; do
    echo "Processing $file..."
    python -m gPLB "$file" -G2
done

# Phase 2: Visualize all with consistent settings
for gml in *.gml; do
    python draw_lattice.py "$gml" \
        --layout multipartite \
        --fig_size 12,12 \
        --fig_dpi 300
done
```

### Example 3: Finding Optimal Z-Score Thresholds

```bash
# Build once
python -m gPLB data.txt -G3

# Test different thresholds
for lb in -3 -2 -1 0; do
    for ub in 1 2 3 4; do
        python draw_lattice.py g3PL-data.gml \
            --zscore_lb $lb --zscore_ub $ub \
            -o "threshold_${lb}_${ub}.png"
    done
done

# Review all images to find optimal threshold
```

### Example 4: Multiple Datasets, Multiple Visualizations

```bash
# Build lattices for all datasets
for dataset in dataset1.txt dataset2.txt dataset3.txt; do
    python -m gPLB "$dataset" -G3
done

# Create standard view for all
for gml in *.gml; do
    python draw_lattice.py "$gml" --layout multipartite -o "${gml%.gml}_standard.png"
done

# Create alternative view for all
for gml in *.gml; do
    python draw_lattice.py "$gml" --layout spring -o "${gml%.gml}_spring.png"
done

# Create filtered view for all
for gml in *.gml; do
    python draw_lattice.py "$gml" --zscore_lb 0 -o "${gml%.gml}_filtered.png"
done
```

### Example 5: Development Workflow

```bash
# Build test lattice once
python -m gPLB test.txt -G3 -n 100

# Rapid iteration on visualization code
while true; do
    # Modify draw_lattice.py
    vim draw_lattice.py
    
    # Test immediately (no rebuild!)
    python draw_lattice.py g3PL-test.gml --show
done
```

### Example 6: Publication Figures

```bash
# Build final lattice
python -m gPLB final_data.txt -G3

# Main figure (multipartite layout)
python draw_lattice.py g3PL-final_data.gml \
    --layout multipartite \
    --fig_size 20,16 \
    --fig_dpi 600 \
    --zscore_lb -1 --zscore_ub 2 \
    -o Figure1_lattice.png

# Supplementary figure (alternative layout)
python draw_lattice.py g3PL-final_data.gml \
    --layout spring \
    --fig_size 15,15 \
    --fig_dpi 300 \
    -o FigureS1_alternative.png

# Supplementary figure (filtered)
python draw_lattice.py g3PL-final_data.gml \
    --layout multipartite \
    --zscore_lb 0 \
    --fig_size 15,15 \
    --fig_dpi 300 \
    -o FigureS2_high_productivity.png
```

---

## GML Format Details

### What's in a GML File?

GML files contain:
- **Nodes**: Pattern forms, ranks, gap sizes, z-scores
- **Edges**: Links between patterns with types
- **Metadata**: Generality level, metric, gap mark

### Example GML Structure

```xml
<?xml version="1.0" encoding="UTF-8"?>
<graphml>
  <graph edgedefault="directed">
    <!-- Metadata -->
    <data key="generality">3</data>
    <data key="p_metric">rank</data>
    
    <!-- Node -->
    <node id="('a', 'b', 'c')">
      <data key="label">a b c</data>
      <data key="rank">3</data>
      <data key="gap_size">0</data>
      <data key="source_zscore">2.45</data>
    </node>
    
    <!-- Edge -->
    <edge source="('_', 'a', 'b', 'c')" target="('a', 'b', 'c')">
      <data key="link_type">instantiates</data>
    </edge>
  </graph>
</graphml>
```

### Advantages of GML

1. **Human-readable**: Open in text editor to inspect
2. **Standard format**: Works with many graph tools
3. **Cross-platform**: Same file works on Windows/Mac/Linux
4. **Version-independent**: Not tied to Python/pickle version
5. **Shareable**: Send to colleagues, they can visualize

---

## Using GML Files in Other Tools

### Gephi (Interactive Visualization)

```bash
# 1. Generate GML
python -m gPLB data.txt -G3

# 2. Open Gephi
# 3. File → Open → Select g3PL-data.gml
# 4. Use Gephi's interactive layout and filtering
```

### Cytoscape (Network Analysis)

```bash
# 1. Generate GML
python -m gPLB data.txt -G3

# 2. Open Cytoscape
# 3. File → Import → Network from File
# 4. Analyze with Cytoscape's network analysis tools
```

### R with igraph

```r
library(igraph)

# Read GML
g <- read_graph("g3PL-data.gml", format="gml")

# Analyze
print(vcount(g))  # Number of nodes
print(ecount(g))  # Number of edges

# Visualize
plot(g, vertex.size=5, edge.arrow.size=0.3)
```

### Python NetworkX

```python
import networkx as nx
import matplotlib.pyplot as plt

# Read GML
G = nx.read_gml("g3PL-data.gml")

# Analyze
print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")
print(f"Generality: {G.graph['generality']}")

# Custom analysis
degrees = dict(G.degree())
high_degree = [n for n, d in degrees.items() if d > 5]
print(f"High-degree nodes: {len(high_degree)}")

# Custom visualization
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True)
plt.savefig("custom_viz.png")
```

---

## Performance Comparison

### Old Workflow (Drawing Integrated)

```
Build + Draw:     30 minutes
Change layout:    30 minutes (rebuild)
Change z-score:   30 minutes (rebuild)
Change size:      30 minutes (rebuild)
Total:            120 minutes
```

### New Workflow (GML-Based)

```
Build + Save GML: 30 minutes
Draw #1:          10 seconds
Draw #2:          10 seconds
Draw #3:          10 seconds
Draw #4:          10 seconds
Total:            31 minutes
```

**Savings: 74% for 4 visualizations!**

---

## Migration Guide

### If You Have Existing Scripts

**Option 1**: Add `-D` to existing calls
```bash
# Old
python -m gPLB data.txt -G3

# New (same behavior)
python -m gPLB data.txt -G3 -D
```

**Option 2**: Update to new workflow
```bash
# Old script
python -m gPLB data1.txt -G3
python -m gPLB data2.txt -G3

# New script (faster!)
# Build phase
python -m gPLB data1.txt -G3
python -m gPLB data2.txt -G3

# Draw phase
python draw_lattice.py g3PL-data1.gml
python draw_lattice.py g3PL-data2.gml
```

**Option 3**: Batch find-and-replace
```bash
# Add -D flag to all scripts
find . -name "*.sh" -exec sed -i 's/python -m gPLB/python -m gPLB -D/g' {} +
```

---

## Troubleshooting

### "Module 'networkx' not found"

```bash
pip install networkx
```

### "GML file has no nodes"

Check if lattice was built successfully:
```bash
python draw_lattice.py file.gml --inspect
```

### "Can't read GML in Gephi"

Make sure you're using NetworkX to save:
```python
import networkx as nx
G = nx.DiGraph()
# ... add nodes/edges ...
nx.write_gml(G, "output.gml")
```

### "Drawing is very slow"

For large lattices:
1. Filter by z-score first: `--zscore_lb 0`
2. Use simpler layout: `--layout spring`
3. Reduce node size: `--node_size 100`

### "Want old behavior back"

Add `-D --no_gml` flags:
```bash
python -m gPLB data.txt -G3 -D --no_gml
```

---

## Tips and Best Practices

### 1. Organize GML Files

```bash
mkdir gml_files
python -m gPLB data.txt -G3 -g gml_files/data.gml
```

### 2. Version Control GML Files

GML files are text-based, so they work well with git:
```bash
git add *.gml
git commit -m "Add lattice GML files"
```

### 3. Document Your Visualizations

```bash
# Create visualization log
echo "$(date): Created multipartite view" >> viz_log.txt
python draw_lattice.py data.gml --layout multipartite
```

### 4. Automate with Makefiles

```makefile
# Makefile
%.gml: %.txt
	python -m gPLB $< -G3 -g $@

%_viz.png: %.gml
	python draw_lattice.py $< -o $@

all: data1_viz.png data2_viz.png data3_viz.png
```

Usage: `make all`

### 5. Compare Visualizations

```bash
# Create all variants
python draw_lattice.py data.gml --layout multipartite -o data_mp.png
python draw_lattice.py data.gml --layout spring -o data_spring.png
python draw_lattice.py data.gml --layout kamada_kawai -o data_kk.png

# View side-by-side
montage data_*.png -geometry 800x600 comparison.png
```

---

## Summary

| Feature | Old | New |
|---------|-----|-----|
| Default behavior | Build + Draw | Build + Save GML |
| Drawing | Automatic | Manual with `-D` or `draw_lattice.py` |
| Visualization changes | Rebuild (30 min) | Redraw (10 sec) |
| File format | PNG only | GML (standard) + PNG |
| Tool compatibility | None | Gephi, Cytoscape, igraph |
| Batch processing | Slow | Fast |

**The new GML-based workflow makes gPLB faster, more flexible, and compatible with the broader graph analysis ecosystem!**

---

## Files in This Package

1. **gml_support.py** - GML save/load functions
2. **draw_lattice.py** - Standalone drawing script
3. **GML_PATCHES.py** - Code patches for __main__.py
4. **GML_WORKFLOW_GUIDE.md** - This guide

---

## Next Steps

1. Apply patches from `GML_PATCHES.py`
2. Test with small dataset: `python -m gPLB test.txt -G2`
3. Verify GML created: `ls *.gml`
4. Test drawing: `python draw_lattice.py g2PL-test.gml`
5. Update your scripts to new workflow
6. Enjoy faster, more flexible lattice visualization! 🎉
