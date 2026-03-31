#!/bin/bash
### USEAGE: ./GIT_ALL.sh "Commit message-check"

msg="${1:-Update repositories}"

repos=(
  "/home/darshan/A6/PCSAFT_cDFT/thermoift/src/thermoift/BINARY_INTERACTION_PARAMETERS"
  "/home/darshan/A6/PCSAFT_cDFT/thermoift"
  "/home/darshan/A6"
)

for repo in "${repos[@]}"; do
  echo
  echo "======================================"
  echo "Processing: $repo"
  echo "======================================"

  cd "$repo" || { echo "Cannot enter $repo"; continue; }

  if [ ! -d .git ] && [ ! -f .git ]; then
    echo "Not a git repo"
    continue
  fi

  branch=$(git branch --show-current)
  [ -z "$branch" ] && branch="master"

  git add -A

  if ! git diff --cached --quiet; then
    git commit -m "$msg ($(basename "$repo"))"
  else
    echo "No changes to commit"
  fi

  git push -u origin "$branch" || echo "Push failed in $repo"
done