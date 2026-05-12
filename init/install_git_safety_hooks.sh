#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
hook_src="$repo_root/init/git-hooks/pre-commit"
hook_dst="$repo_root/.git/hooks/pre-commit"

if [[ ! -f "$hook_src" ]]; then
  echo "Missing hook source: $hook_src" >&2
  exit 1
fi

mkdir -p "$repo_root/.git/hooks"

if [[ -f "$hook_dst" ]]; then
  backup="$hook_dst.backup.$(date +%Y%m%d%H%M%S)"
  cp "$hook_dst" "$backup"
  echo "Backed up existing pre-commit hook to: $backup"
fi

cp "$hook_src" "$hook_dst"
chmod +x "$hook_dst"

echo "Installed Git safety pre-commit hook: $hook_dst"
echo "The hook blocks staged large files and common model/artifact extensions."
