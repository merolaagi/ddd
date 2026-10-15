#!/usr/bin/env bash
set -e
SRC="$(cd "$(dirname "$0")" && pwd)"
DEST="${DDD_DEST:-$HOME/Sites/ddd}"
REPO="${DDD_REPO:-merolaagi/ddd}"
VERSION="$(cat "$SRC/VERSION")"
mkdir -p "$DEST"
if [ -d "$DEST/.git" ]; then
  (cd "$SRC" && find . -type f ! -path "./.git/*" | sed 's|^\./||' | sort) > /tmp/ddd-release-files.$$
  (cd "$DEST" && git ls-files) | while read -r f; do
    if ! grep -qxF "$f" /tmp/ddd-release-files.$$; then
      (cd "$DEST" && git rm -q --cached -- "$f" && rm -f -- "$f") && echo "Removed (no longer in release): $f"
    fi
  done
  rm -f /tmp/ddd-release-files.$$
fi
(cd "$SRC" && tar --exclude ./.venv --exclude ./data --exclude ./logs --exclude ./.env --exclude ./.git -cf - .) | (cd "$DEST" && tar -xf -)
cd "$DEST"
chmod +x serve.sh install.sh
if [ ! -d .venv ]; then
  PY=$(command -v python3.12 || command -v python3.13 || command -v python3.11 || command -v python3)
  "$PY" -m venv .venv
fi
echo "Installing Python packages ..."
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt
echo "Installing headless Chromium for the Tools Armory (first time only) ..."
.venv/bin/python -m playwright install chromium > /dev/null || echo "Warning: Chromium download failed; rerun .venv/bin/python -m playwright install chromium before using recon."
[ -f .env ] || cp .env.example .env
command -v npx > /dev/null || echo "Note: REA needs Node.js 22+ (brew install node). Everything else works without it."
if [ ! -d .git ]; then git init -q -b main; fi
if ! git remote get-url origin > /dev/null 2>&1; then
  if command -v gh > /dev/null && gh auth status > /dev/null 2>&1; then
    gh repo view "$REPO" > /dev/null 2>&1 || gh repo create "$REPO" --public --description "ddd: a headless drug-discovery intelligence company run by agents" > /dev/null
  fi
  git remote add origin "https://github.com/$REPO.git"
fi
git add -A
git commit -q -m "ddd v$VERSION" || echo "No changes to commit."
echo "Installed ddd v$VERSION into $DEST"
