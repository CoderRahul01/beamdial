#!/usr/bin/env bash
# One command launch, from your own Mac terminal:
#
#   export HF_TOKEN=...        # write access, never paste this line anywhere
#   bash scripts/launch.sh             # public
#   bash scripts/launch.sh --private   # private first, to review; rerun without the flag to go public
#
# Needs: GitHub CLI (brew install gh; gh auth login) and huggingface_hub (pip install -U huggingface_hub).
set -euo pipefail
cd "$(dirname "$0")/.."
VIS_FLAG="${1:-}"

command -v gh >/dev/null || { echo "Install the GitHub CLI: brew install gh, then gh auth login"; exit 1; }
gh auth status >/dev/null 2>&1 || gh auth login
GH_OWNER="$(gh api user -q .login)"
python3 -c "import huggingface_hub" 2>/dev/null || python3 -m pip install -q -U huggingface_hub
[ -n "${HF_TOKEN:-}" ] || { echo "export HF_TOKEN=... (write access) first"; exit 1; }

echo "== 1/3 tests"
python3 -m pip install -q pytest >/dev/null 2>&1 || true
python3 -m pytest -q tests

echo "== 2/3 Hugging Face dataset + Space"
python3 scripts/publish_hf.py --github-owner "$GH_OWNER" ${VIS_FLAG}

echo "== 3/3 GitHub"
if [ "$VIS_FLAG" = "--private" ]; then
  bash scripts/publish_github.sh "$GH_OWNER" private
else
  bash scripts/publish_github.sh "$GH_OWNER" public
fi

HF_USER="$(python3 -c "from huggingface_hub import HfApi; import os; print(HfApi(token=os.environ['HF_TOKEN']).whoami()['name'])")"
echo
echo "Launched:"
echo "  GitHub:  https://github.com/$GH_OWNER/beamdial"
echo "  Dataset: https://huggingface.co/datasets/$HF_USER/beamdial-v0.1"
echo "  Space:   https://huggingface.co/spaces/$HF_USER/beamdial"
echo "Next: post the submission comment from private/launch-and-marketing.md with these links."
