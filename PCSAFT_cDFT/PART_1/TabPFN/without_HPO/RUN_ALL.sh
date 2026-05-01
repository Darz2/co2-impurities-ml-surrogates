#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

scripts=(
    "RUNBubble.sh"
    "RUNDew.sh"
    "RUNGamma.sh"
    "RUNThickness.sh"
)

TEST_ROWS="${TEST_ROWS:-None}"

echo "Submitting TabPFN jobs from: ${SCRIPT_DIR}"
echo "TEST_ROWS=${TEST_ROWS}"

for script in "${scripts[@]}"; do
    if [[ ! -x "${script}" ]]; then
        echo "Making ${script} executable"
        chmod +x "${script}"
    fi

    sed -i 's/^#SBATCH --partition=.*/#SBATCH --partition=highmem/' "${script}"

    echo "Submitting ${script}"
    sbatch --export=ALL,TEST_ROWS="${TEST_ROWS}" "${script}"
done

echo "All TabPFN jobs submitted."
