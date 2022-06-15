from pathlib import Path

import git


def get_repo_root():
    """
    Returns the root path of current git repo.
    """
    repo = git.Repo(__file__, search_parent_directories=True)
    return Path(repo.working_tree_dir)


TMP_DIR = get_repo_root() / "tmp"
TMP_DIR.mkdir(exist_ok=True)

REPO_ROOT = get_repo_root()

EXPERIMENTS_DIR = REPO_ROOT / "experiments"
EXPERIMENTS_DIR.mkdir(exist_ok=True)

DATASETS_DIR = REPO_ROOT / "datasets"
DATASETS_DIR.mkdir(exist_ok=True)
