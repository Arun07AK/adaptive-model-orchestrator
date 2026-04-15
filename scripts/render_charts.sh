#!/bin/bash
# Render HTML chart templates to PNG via Chrome headless.
# Output: 1080x1080 PNGs in data/charts/

set -e

CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHARTS_DIR="$PROJECT_ROOT/scripts/charts"
OUT_DIR="$PROJECT_ROOT/data/charts"

mkdir -p "$OUT_DIR"

render() {
    local html_file="$1"
    local png_name="$2"
    echo "  Rendering $png_name..."
    "$CHROME" \
        --headless=new \
        --disable-gpu \
        --no-sandbox \
        --hide-scrollbars \
        --force-device-scale-factor=2 \
        --window-size=1080,1080 \
        --screenshot="$OUT_DIR/$png_name" \
        "file://$CHARTS_DIR/$html_file" \
        2>/dev/null
}

echo "Rendering LinkedIn carousel images..."
render "01_hero.html"         "01_hero.png"
render "02_benchmarks.html"   "02_accuracy.png"
render "03_architecture.html" "03_architecture.png"
render "04_research.html"     "04_research.png"

echo ""
echo "Done. Output files:"
ls -la "$OUT_DIR"/*.png
