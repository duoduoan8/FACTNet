#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

REPO_NAME="${1:-FACTNet}"
VISIBILITY="${2:-public}"  # public|private
REMOTE_URL="${3:-}"         # optional, e.g. https://github.com/<user>/<repo>.git

if ! command -v git >/dev/null 2>&1; then
  echo "[ERROR] git is not installed."
  exit 1
fi

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "[ERROR] Not a git repository: $ROOT_DIR"
  echo "Run: git init && git add . && git commit -m 'Initial commit'"
  exit 1
fi

git branch -M main >/dev/null 2>&1 || true

case "$VISIBILITY" in
  public)
    VIS_FLAG="--public"
    ;;
  private)
    VIS_FLAG="--private"
    ;;
  *)
    echo "[ERROR] VISIBILITY must be 'public' or 'private' (got: $VISIBILITY)"
    exit 1
    ;;
esac

if git remote get-url origin >/dev/null 2>&1; then
  echo "[FACTNet] Remote 'origin' already set: $(git remote get-url origin)"
  echo "[FACTNet] Pushing to origin/main..."
  git push -u origin main
  exit 0
fi

if [[ -n "$REMOTE_URL" ]]; then
  echo "[FACTNet] Setting remote 'origin' to: $REMOTE_URL"
  git remote add origin "$REMOTE_URL"
  echo "[FACTNet] Pushing to origin/main..."
  if ! git push -u origin main; then
    echo "[ERROR] git push failed (likely authentication)."
    echo "Login options:"
    echo "  - gh auth login"
    echo "  - or configure a GitHub Personal Access Token for HTTPS"
    exit 1
  fi
  exit 0
fi

if ! command -v gh >/dev/null 2>&1; then
  echo "[ERROR] GitHub CLI (gh) is not installed."
  echo "Install it from: https://cli.github.com/"
  echo "Or re-run with an existing repo URL as the 3rd argument."
  exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
  echo "[ERROR] You are not logged into GitHub via gh."
  echo "Run: gh auth login"
  exit 1
fi

echo "[FACTNet] Creating GitHub repo '$REPO_NAME' ($VISIBILITY) and pushing..."
gh repo create "$REPO_NAME" $VIS_FLAG --source=. --remote=origin --push
