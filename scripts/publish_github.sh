#!/usr/bin/env bash
# Publish BeamDial to GitHub from your own Mac terminal.
#
#   bash scripts/publish_github.sh                 # public repo under your GitHub user
#   bash scripts/publish_github.sh my-org private  # under an org, private
#
# Needs the GitHub CLI: brew install gh && gh auth login
set -euo pipefail
cd "$(dirname "$0")/.."

command -v gh >/dev/null || { echo "Install the GitHub CLI first: brew install gh, then: gh auth login"; exit 1; }
gh auth status >/dev/null 2>&1 || gh auth login

OWNER="${1:-$(gh api user -q .login)}"
VISIBILITY="${2:-public}"
REPO="$OWNER/beamdial"
echo "Publishing to github.com/$REPO ($VISIBILITY)"

# 1. Safety: stop if anything that looks like a secret is about to be committed.
if grep -rInE "hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{30,}" --exclude-dir=.git --exclude-dir=private --exclude-dir=_archive --exclude-dir=data . ; then
  echo "Found something that looks like an API token (above). Remove it, then run this again."
  exit 1
fi

# 2. Fill the OWNER placeholder in links and badges.
{ grep -rl "OWNER/beamdial" README.md docs space_static 2>/dev/null || true; } | while read -r f; do
  perl -pi -e "s#OWNER/beamdial#$REPO#g" "$f"
done

# 3. Commit.
[ -d .git ] || git init -b main
git add -A
git status --short
git commit -m "BeamDial v0.1.0: pick Beam's reasoning_effort per request

Router, smart escalation, production fixes for truncation, tool calls and
reasoning_content, plus the measurement pipeline, probe, Space and docs.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017xQf2Y6Syih2rotMAMduUT" || echo "(nothing new to commit)"

# 4. Create the repo and push (or just push if it already exists).
if gh repo view "$REPO" >/dev/null 2>&1; then
  git remote get-url origin >/dev/null 2>&1 || git remote add origin "https://github.com/$REPO.git"
  git push -u origin main
  gh repo edit "$REPO" --visibility "$VISIBILITY" --accept-visibility-change-consequences
else
  gh repo create "$REPO" "--$VISIBILITY" --source . --remote origin --push \
    --description "Pick Beam's reasoning_effort per request. Pay for max thinking only when it helps."
fi

# 5. Topics so people can find it.
gh repo edit "$REPO" --add-topic beam,reflection-ai,huggingface,llm,reasoning-models,llm-router,test-time-compute,cost-optimization,inference,openai-compatible,ai-agents,benchmark,python

# 6. Tagged release, only when public and not already released.
VERSION="v$(python3 -c "import re;print(re.search(r'version = \"([^\"]+)\"', open('pyproject.toml').read()).group(1))")"
if [ "$VISIBILITY" = "public" ] && ! gh release view "$VERSION" --repo "$REPO" >/dev/null 2>&1; then
  gh release create "$VERSION" --repo "$REPO" --title "BeamDial $VERSION" --notes-file .github/release-notes.md
fi

echo "Done: https://github.com/$REPO"
