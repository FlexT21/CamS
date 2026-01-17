from pathlib import Path


def project_root(anchor: str = "pyproject.toml"):
    path = Path.cwd()
    for parent in [path] + list(path.parents):
        if (parent / anchor).exists():
            return parent
    raise FileNotFoundError(
        f"Could not find {anchor} in the parent directories of {path}"
    )


ROOTDIR = project_root()
SRCDIR = ROOTDIR / "src"
USERSDIR = ROOTDIR / "users"

if __name__ == "__main__":
    import sys

    sys.stdout.write(f"{ROOTDIR}\n")
