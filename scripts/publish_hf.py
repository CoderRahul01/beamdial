"""Publish the BeamDial dataset and Space to your Hugging Face account. Run on your Mac.

    pip install -U huggingface_hub
    export HF_TOKEN=...            # a token with WRITE access (huggingface.co/settings/tokens)
    python3 scripts/publish_hf.py --github-owner <your-github-user>

Creates (or updates):
  datasets/<you>/beamdial-v0.1   runs, tasks, flagged tasks, summary, spreadsheet, figures
  spaces/<you>/beamdial          the demo app
"""
import argparse
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TPL = ROOT / "hub_templates"

DATASET_FILES = {
    "data/tasks.jsonl": "data/tasks.jsonl",
    "data/graded.jsonl": "data/runs_graded.jsonl",
    "data/flagged_tasks.json": "data/flagged_tasks.json",
    "results/summary.json": "results/summary.json",
    "results/report.md": "results/report.md",
    "results/BeamDial-results-v0.1.xlsx": "results/BeamDial-results-v0.1.xlsx",
    "results/launch_figures.json": "results/launch_figures.json",
}
SPACE_FILES = {  # static Space: free to host, the router runs in the visitor's browser
    "space_static/index.html": "index.html",
    "space_static/router.js": "router.js",
    "beamdial/router_weights.json": "router_weights.json",
    "results/summary.json": "summary.json",
    "assets/figures/01-hero.png": "figures/01-hero.png",
    "assets/figures/02-dial-works.png": "figures/02-dial-works.png",
    "assets/figures/03-policies.png": "figures/03-policies.png",
    "assets/figures/04-long-think-signal.png": "figures/04-long-think-signal.png",
}
SECRET = re.compile(r"hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{30,}")


def fill(text, hf_user, gh_owner):
    return text.replace("HFUSER/", f"{hf_user}/").replace("OWNER/", f"{gh_owner}/")


def stage(files, extra_dirs, readme_tpl, hf_user, gh_owner, extra_text=None):
    d = Path(tempfile.mkdtemp(prefix="beamdial_hub_"))
    for src, dst in files.items():
        p = ROOT / src
        if not p.exists():
            sys.exit(f"Missing {src}. Run the pipeline first (grade, train_router, analyze, make_figures, make_sheet).")
        (d / dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, d / dst)
    for src, dst in extra_dirs.items():
        shutil.copytree(ROOT / src, d / dst, ignore=shutil.ignore_patterns("__pycache__", "*.html"))
    (d / "README.md").write_text(fill((TPL / readme_tpl).read_text(), hf_user, gh_owner))
    for name, text in (extra_text or {}).items():
        (d / name).write_text(text)
    for f in d.rglob("*"):  # last line of defense against leaking a token
        if f.is_file() and f.suffix in (".py", ".md", ".json", ".jsonl", ".txt", ".html", ".js") and SECRET.search(f.read_text(errors="ignore")):
            sys.exit(f"Refusing to upload: {f.name} contains something that looks like an API token.")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--github-owner", required=True, help="your GitHub username or org (for links)")
    ap.add_argument("--dataset-name", default="beamdial-v0.1")
    ap.add_argument("--space-name", default="beamdial")
    ap.add_argument("--private", action="store_true", help="create both repos as private (to review before going public)")
    args = ap.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("Run: pip install -U huggingface_hub")
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("Set HF_TOKEN to a token with write access first.")
    api = HfApi(token=token)
    me = api.whoami()["name"]
    print(f"Hugging Face user: {me}")

    ds_id, sp_id = f"{me}/{args.dataset_name}", f"{me}/{args.space_name}"
    license_txt = ("Prompts and reference answers come from GSM8K, MATH, AMC 2023, AIME 2024, OlympiadBench, HumanEval and MBPP "
                   "and keep their original licenses and terms.\nBeam outputs, grades, annotations and code: Apache License 2.0.\n")

    ds_dir = stage(DATASET_FILES, {"assets/figures": "figures"}, "dataset_README.md", me, args.github_owner, {"LICENSE": license_txt})
    for f in (ds_dir / "figures").glob("*.html"):
        f.unlink()
    api.create_repo(ds_id, repo_type="dataset", private=args.private, exist_ok=True)
    api.upload_folder(repo_id=ds_id, repo_type="dataset", folder_path=str(ds_dir),
                      commit_message="BeamDial v0.1: 2,050 graded runs across Beam's 5 reasoning levels")
    print(f"Dataset: https://huggingface.co/datasets/{ds_id}")

    sp_dir = stage(SPACE_FILES, {}, "space_README.md", me, args.github_owner)
    page = sp_dir / "index.html"
    page.write_text(fill(page.read_text(), me, args.github_owner))
    api.create_repo(sp_id, repo_type="space", space_sdk="static", private=args.private, exist_ok=True)
    api.upload_folder(repo_id=sp_id, repo_type="space", folder_path=str(sp_dir),
                      commit_message="BeamDial demo: pick Beam's reasoning level per prompt")
    print(f"Space:   https://huggingface.co/spaces/{sp_id}  (static, free; live in under a minute)")

    # make visibility match the flag, also for repos created earlier (private review first, public later)
    for rid, rtype in ((ds_id, "dataset"), (sp_id, "space")):
        try:
            api.update_repo_settings(repo_id=rid, repo_type=rtype, private=args.private)
        except AttributeError:  # older huggingface_hub
            api.update_repo_visibility(repo_id=rid, repo_type=rtype, private=args.private)
    print("Visibility:", "PRIVATE (only you can see them)" if args.private else "PUBLIC")

    # point the GitHub README at the real Hub links
    for p in [ROOT / "README.md", *(ROOT / "docs").glob("*.md")]:
        t = p.read_text()
        nt = fill(t, me, args.github_owner)
        if nt != t:
            p.write_text(nt)
    print("Updated README/docs links. Next: bash scripts/publish_github.sh")


if __name__ == "__main__":
    main()
