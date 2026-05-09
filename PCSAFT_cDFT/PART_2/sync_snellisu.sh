#!/bin/bash

REMOTE_HOST="snellius"
REMOTE_BASE="/scratch-shared/draju/PART_2"
BASE_LOCAL="/home/darshan/A6/PCSAFT_cDFT/PART_2"

sync_dir() {
    local NAME="$1"
    local REMOTE_PATH="$REMOTE_BASE/$2"
    local LOCAL_DIR="$BASE_LOCAL/$3"

    echo "=== Syncing $NAME ==="
    REMOTE_COUNT=$(ssh "$REMOTE_HOST" "ls $REMOTE_PATH 2>/dev/null | wc -l")
    if [ "$REMOTE_COUNT" -eq 0 ]; then
        echo "  Remote empty — pushing local -> remote"
        rsync -avz --progress "$LOCAL_DIR/" "$REMOTE_HOST:$REMOTE_PATH/"
    else
        echo "  Remote has files — pulling remote -> local"
        rsync -avz --progress "$REMOTE_HOST:$REMOTE_PATH/" "$LOCAL_DIR/"
    fi
}

push_combined() {
    local NAME="$1"
    local REMOTE_PATH="$REMOTE_BASE/$2/COMBINED"
    local LOCAL_DIR="$BASE_LOCAL/$3/COMBINED"

    echo "=== Push clean COMBINED: $NAME ==="
    echo "  Deleting remote $REMOTE_PATH ..."
    ssh "$REMOTE_HOST" "rm -rf '$REMOTE_PATH'"
    echo "  Pushing local -> remote ..."
    rsync -avz --progress "$LOCAL_DIR/" "$REMOTE_HOST:$REMOTE_PATH/"
    echo "  Done."
}

push_combined "RANDOM"        "RANDOM"        "RANDOM"
push_combined "STRATIFIED"    "STRATIFIED"    "STRATIFIED"
push_combined "ACTIVELEARNING" "ACTIVELEARNING" "ActiveLearning"

sync_dir "ACTIVELEARNING/AL_MT" "ACTIVELEARNING/AL_MT" "ActiveLearning/AL_MT"
sync_dir "ACTIVELEARNING/AL_ST" "ACTIVELEARNING/AL_ST" "ActiveLearning/AL_ST"
