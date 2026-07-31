#!/usr/bin/env bash
# Build PDFs for all bundle papers (Papers 1–4).
# Usage: bash build_pdfs.sh [--with-contributors] [paper...]
#   No args              — builds all four papers
#   --with-contributors  — inject contributors.md into acknowledgements
#   With paper names     — builds named papers only: framework gem validation prh
#
# Requires: pandoc, xelatex, DejaVu fonts

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PANDOC_OPTS=(
    --pdf-engine=xelatex
    -V geometry:margin=1in
    -V fontsize=11pt
    -V mainfont="DejaVu Serif"
    -V monofont="DejaVu Sans Mono"
    --include-in-header="$SCRIPT_DIR/shared/figure-placement.tex"
    --standalone
)

WITH_CONTRIBUTORS=0

build() {
    local name="$1"
    local dir="$2"
    echo "Building $name..."
    cd "$SCRIPT_DIR/$dir"

    local src="preprint.md"
    local tmp=""

    if [[ "$WITH_CONTRIBUTORS" == "1" ]] && [[ -f contributors.md ]]; then
        tmp=$(mktemp /tmp/preprint_XXXXXX.md)
        sed "/<!-- contributors -->/r contributors.md" preprint.md \
            | sed "/<!-- contributors -->/d" > "$tmp"
        src="$tmp"
    fi

    pandoc "$src" -o preprint.pdf "${PANDOC_OPTS[@]}" 2>&1 \
        | grep -iv "^$\|missing character" || true

    [[ -n "$tmp" ]] && rm -f "$tmp"
    echo "  → $dir/preprint.pdf"

    # Papers with a companion supplement (P3, P4) build it to supplementary.pdf.
    if [[ -f supplementary.md ]]; then
        pandoc supplementary.md -o supplementary.pdf "${PANDOC_OPTS[@]}" 2>&1 \
            | grep -iv "^$\|missing character" || true
        echo "  → $dir/supplementary.pdf"
    fi
}

declare -A PAPERS=(
    [framework]="caz-framework"
    [gem]="gem"
    [validation]="caz-validation"
    [prh]="prh-validation"
)

# Parse flags and targets
targets=()
for arg in "$@"; do
    if [[ "$arg" == "--with-contributors" ]]; then
        WITH_CONTRIBUTORS=1
    elif [[ -n "${PAPERS[$arg]+x}" ]]; then
        targets+=("$arg")
    else
        echo "Unknown argument '$arg'. Valid papers: ${!PAPERS[*]}" >&2
        exit 1
    fi
done

if [[ ${#targets[@]} -eq 0 ]]; then
    targets=(framework gem validation prh)
fi

for t in "${targets[@]}"; do
    build "$t" "${PAPERS[$t]}"
done

echo "Done."
