# safe_gplb.sh Usage Guide

## Output Format

**By default, gPLB now produces `.gml` (Graph Modeling Language) files** instead of drawing lattices. This is more efficient and allows you to visualize the results with your preferred graph visualization tool.

- **To get .gml output only** (default): Run without -D flag
- **To also generate a lattice diagram**: Add the `-D` flag
- **Figure customization** (--fig_size, --fig_dpi): Only applies when using `-D` flag

## Two Modes Available

### Mode 1: Adaptive (Automatic)
The script analyzes your data and chooses the best generality level automatically.

### Mode 2: Fixed (Pre-defined)
You specify the exact generality level you want to use.

---

## Usage Examples

### Adaptive Mode (Automatic G selection)

```bash
# Let script decide everything
./safe_gplb.sh data.txt

# Explicit adaptive mode
./safe_gplb.sh data.txt auto

# Adaptive with diagram output
./safe_gplb.sh data.txt auto -D

# Adaptive with additional options
./safe_gplb.sh data.txt auto -v -D
```

**What happens:** Script calculates average segment length and chooses:
- G3 for segments ≤8 words
- G2 for segments 9-10 words
- G1 for segments 11-15 words
- G0 for segments >15 words

**Output:** Creates a `.gml` file (add `-D` flag to also generate a lattice diagram)

---

### Fixed Mode (Pre-defined G level)

```bash
# Force G3 (always use highest generality)
./safe_gplb.sh data.txt 3

# Force G2
./safe_gplb.sh data.txt 2

# Force G1
./safe_gplb.sh data.txt 1

# Force G0 (no generalization)
./safe_gplb.sh data.txt 0

# Fixed G3 with verbose output
./safe_gplb.sh data.txt 3 -v

# Fixed G2 with diagram output and custom figure size (requires -D for diagram)
./safe_gplb.sh data.txt 2 -D --fig_size 12,12 --fig_dpi 600

# Fixed G1 with multiple options including diagram
./safe_gplb.sh data.txt 1 -v -D -I
```

**What happens:** Script uses your specified G level but still applies smart sampling based on segment length to prevent memory issues.

**Output:** Creates a `.gml` file (add `-D` flag to also generate a lattice diagram)

---

## Understanding Fixed Mode Behavior

When you specify a fixed generality level, the script still protects you from memory issues by adjusting sampling:

### Fixed G3 Behavior:
```
Segment Length  →  Sampling Applied
≤5 words        →  No sampling (safe)
6-8 words       →  -n 1000 (safe)
9-10 words      →  -n 500 (warning: may use lots of memory)
>10 words       →  -n 200 (warning: HIGH MEMORY RISK)
```

### Fixed G2 Behavior:
```
Segment Length  →  Sampling Applied
≤8 words        →  No sampling (safe)
9-12 words      →  -n 800 (safe)
>12 words       →  -n 300 (safe)
```

### Fixed G1 Behavior:
```
Segment Length  →  Sampling Applied
≤15 words       →  No sampling (safe)
>15 words       →  -n 500 (safe)
```

### Fixed G0 Behavior:
```
Any length      →  No sampling (always safe)
```

---

## Quick Reference Table

| Command | Mode | Generality | Sampling | Use Case |
|---------|------|------------|----------|----------|
| `./safe_gplb.sh data.txt` | Adaptive | Auto | Auto | Safest, recommended for most cases |
| `./safe_gplb.sh data.txt 3` | Fixed | G3 | Auto | Want G3 but with memory protection |
| `./safe_gplb.sh data.txt 2` | Fixed | G2 | Auto | Want G2 specifically |
| `./safe_gplb.sh data.txt 1` | Fixed | G1 | Auto | Want G1 specifically |
| `./safe_gplb.sh data.txt 0` | Fixed | G0 | None | No generalization needed |

---

## Common Scenarios

### Scenario 1: "I always want G3, no matter what"
```bash
./safe_gplb.sh mydata.txt 3
```
✓ Uses G3 but adds sampling for long segments to prevent memory crash

### Scenario 2: "I want maximum safety, let script decide"
```bash
./safe_gplb.sh mydata.txt
```
✓ Fully automatic, chooses both G level and sampling

### Scenario 3: "I want G2 for consistency across datasets"
```bash
./safe_gplb.sh dataset1.txt 2
./safe_gplb.sh dataset2.txt 2
./safe_gplb.sh dataset3.txt 2
```
✓ All use G2, but sampling adjusts per dataset

### Scenario 4: "I need G3 with my own sampling"
```bash
# Script's sampling might not be what you want
# Override by calling gPLB directly:
python -m gPLB mydata.txt -G3 -n 5000
```
✓ Full manual control (but memory risk returns!)

---

## Example Output

### Adaptive Mode:
```
╔════════════════════════════════════════════════════╗
║           Safe gPLB Memory-Aware Wrapper           ║
╚════════════════════════════════════════════════════╝

Input file:      data.txt
Avg seg length:  7 words/segment
Mode:            ADAPTIVE
Selected params: -G3 -n 1000
Reasoning:       Medium-short segments - adaptive G3 with sampling

Starting gPLB...
────────────────────────────────────────────────────
```

### Fixed Mode:
```
╔════════════════════════════════════════════════════╗
║           Safe gPLB Memory-Aware Wrapper           ║
╚════════════════════════════════════════════════════╝

Input file:      data.txt
Avg seg length:  7 words/segment
Mode:            FIXED
Selected params: -G3 -n 1000
Reasoning:       Fixed G3 - medium segments, safe sampling

Starting gPLB...
────────────────────────────────────────────────────
```

---

## Tips

### Tip 1: Compare Adaptive vs Fixed
```bash
# See what adaptive mode would choose
./safe_gplb.sh data.txt auto -v | head -20

# Then decide if you want to override
./safe_gplb.sh data.txt 3 -v
```

### Tip 2: Test Before Large Runs
```bash
# Test with small sample first
head -100 largefile.txt > test.txt
./safe_gplb.sh test.txt 3

# If successful, run full dataset
./safe_gplb.sh largefile.txt 3
```

### Tip 3: Override Sampling If Needed
The script applies sampling, but you can override:
```bash
# Script might choose -n 500, but you want -n 2000
./safe_gplb.sh data.txt 3 -n 2000
```
Note: The later `-n` flag overrides the earlier one in gPLB.

---

## When to Use Each Mode

**Use Adaptive Mode when:**
- You're not sure what generality level is best
- Processing many different datasets
- Want maximum safety
- First time working with the data

**Use Fixed Mode when:**
- You need consistent generality across datasets
- You know exactly what patterns you're looking for
- You're doing comparative analysis (need same G level)
- You understand the memory/pattern tradeoffs

---

## Installation

```bash
# Download script
# (already in your outputs folder)

# Make executable
chmod +x safe_gplb.sh

# Optional: Add to PATH
sudo cp safe_gplb.sh /usr/local/bin/
# Then use from anywhere:
safe_gplb.sh mydata.txt 3
```
