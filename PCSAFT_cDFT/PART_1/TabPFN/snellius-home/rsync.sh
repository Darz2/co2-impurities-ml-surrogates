#!/usr/bin/env bash
set -euo pipefail

REMOTE_DIR="snellius:/home/draju/A6/TabPFN/*"
LOCAL_DIR="/home/darshan/A6/PCSAFT_cDFT/PART_1/TabPFN/snellius-home/"

rsync -avz --progress "${REMOTE_DIR}" "${LOCAL_DIR}/." 
