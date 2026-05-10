#!/usr/bin/env bash
set -euo pipefail

LOCAL_DIR="/home/darshan/A6/PCSAFT_cDFT/PART_1/TabPFN/snellius-home/"
REMOTE_DIR="snellius:/home/draju/A6/TabPFN/*"

rsync -avz --progress "${LOCAL_DIR}/" "${REMOTE_DIR}"
