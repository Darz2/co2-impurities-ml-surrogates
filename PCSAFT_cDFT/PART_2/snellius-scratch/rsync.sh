#!/usr/bin/env bash
set -euo pipefail

LOCAL_DIR="/home/darshan/A6/PCSAFT_cDFT/PART_2/snellius-scratch"
REMOTE_DIR="snellius:/scratch-shared/draju/PART_2/."

rsync -avz --progress "${LOCAL_DIR}/" "${REMOTE_DIR}"
