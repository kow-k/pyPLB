#!/bin/bash
#
# safe_gPLB.sh - Wrapper for gPLB with fixed or adaptive generality
#
# Usage:
#   ./safe_gplb.sh input_file.txt [generality_level] [additional_gplb_options]
#
# Examples:
#   ./safe_gplb.sh data.txt              # Adaptive (automatic)
#   ./safe_gplb.sh data.txt 3            # Fixed G3
#   ./safe_gplb.sh data.txt 2 -v -D      # Fixed G2 with verbose
#   ./safe_gplb.sh data.txt auto         # Explicit adaptive mode

set -e  # Exit on error

# Check if input file provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 input_file [generality_level] [additional_options]"
    echo ""
    echo "generality_level: 0, 1, 2, 3, or 'auto' (default: auto)"
    echo ""
    echo "Examples:"
    echo "  $0 data.txt              # Adaptive mode (auto-select G level)"
    echo "  $0 data.txt 3            # Force G3"
    echo "  $0 data.txt 2 -v         # Force G2 with verbose"
    echo "  $0 data.txt auto -D      # Explicit adaptive mode"
    exit 1
fi

INPUT_FILE="$1"
shift  # Remove first argument

# Check if file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: File '$INPUT_FILE' not found"
    exit 1
fi

# Check if second argument is generality level
FIXED_G=""
if [ $# -ge 1 ]; then
    case "$1" in
        0|1|2|3)
            FIXED_G="$1"
            shift  # Remove generality argument
            ;;
        auto|adaptive|"")
            FIXED_G="auto"
            shift
            ;;
        -*)
            # Starts with dash, it's an option not generality
            FIXED_G="auto"
            ;;
        *)
            echo "Warning: Unknown generality level '$1'. Using adaptive mode."
            FIXED_G="auto"
            shift
            ;;
    esac
else
    FIXED_G="auto"
fi

# Calculate average segment length for adaptive sampling
AVG_LEN=$(awk -F'[,;]' '
    # Skip comments
    /^[#%]/ { next }
    # Skip empty lines
    /^[[:space:]]*$/ { next }
    # Count fields
    {
        sum += NF
        count++
    }
    END {
        if (count > 0)
            print int(sum/count)
        else
            print 5  # Default if no valid lines
    }
' "$INPUT_FILE")

# Determine parameters
if [ "$FIXED_G" = "auto" ]; then
    # Adaptive mode - determine G based on segment length
    if [ "$AVG_LEN" -le 5 ]; then
        G=3
        N=""
        MEMO="Short segments - adaptive G3"
    elif [ "$AVG_LEN" -le 8 ]; then
        G=3
        N="-n 1000"
        MEMO="Medium-short segments - adaptive G3 with sampling"
    elif [ "$AVG_LEN" -le 10 ]; then
        G=2
        N="-n 500"
        MEMO="Medium segments - adaptive G2 with sampling"
    elif [ "$AVG_LEN" -le 15 ]; then
        G=1
        N="-n 200"
        MEMO="Long segments - adaptive G1 with heavy sampling"
    else
        G=0
        N="-n 100"
        MEMO="Very long segments - adaptive G0 only"
    fi
else
    # Fixed generality mode - adjust only sampling
    G=$FIXED_G

    # Determine safe sampling based on G level and segment length
    case $G in
        3)
            if [ "$AVG_LEN" -le 5 ]; then
                N=""
                MEMO="Fixed G3 - short segments, no sampling needed"
            elif [ "$AVG_LEN" -le 8 ]; then
                N="-n 1000"
                MEMO="Fixed G3 - medium segments, safe sampling"
            elif [ "$AVG_LEN" -le 10 ]; then
                N="-n 500"
                MEMO="Fixed G3 - longer segments, heavy sampling (warning: may use lots of memory)"
            else
                N="-n 200"
                MEMO="Fixed G3 - long segments, very heavy sampling (warning: HIGH MEMORY RISK)"
            fi
            ;;
        2)
            if [ "$AVG_LEN" -le 8 ]; then
                N=""
                MEMO="Fixed G2 - segments ok, no sampling needed"
            elif [ "$AVG_LEN" -le 12 ]; then
                N="-n 800"
                MEMO="Fixed G2 - medium-long segments, safe sampling"
            else
                N="-n 300"
                MEMO="Fixed G2 - long segments, heavy sampling"
            fi
            ;;
        1)
            if [ "$AVG_LEN" -le 15 ]; then
                N=""
                MEMO="Fixed G1 - segments ok, no sampling needed"
            else
                N="-n 500"
                MEMO="Fixed G1 - long segments, safe sampling"
            fi
            ;;
        0)
            N=""
            MEMO="Fixed G0 - no generalization, always safe"
            ;;
    esac
fi

# Report settings
echo "╔════════════════════════════════════════════════════╗"
echo "║           Safe gPLB Memory-Aware Wrapper           ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""
echo "Input file:      $INPUT_FILE"
echo "Avg seg length:  $AVG_LEN words/segment"
echo "Mode:            $([ "$FIXED_G" = "auto" ] && echo "ADAPTIVE" || echo "FIXED")"
echo "Selected params: -G$G $N"
echo "Reasoning:       $MEMO"
echo ""
echo "Additional args: $*"
echo ""
echo "Starting gPLB..."
echo "────────────────────────────────────────────────────"

# Build command
CMD="python -m gPLB \"$INPUT_FILE\" -G$G"
[ -n "$N" ] && CMD="$CMD $N"
[ -n "$*" ] && CMD="$CMD $*"

# Execute
echo "Command: $CMD"
echo ""
eval $CMD

# Report completion
echo ""
echo "────────────────────────────────────────────────────"
echo "✓ Complete!"
