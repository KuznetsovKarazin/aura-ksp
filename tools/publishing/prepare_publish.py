"""Verify AURA-KSP manifests and build deterministic ZIPs using Python stdlib."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import zipfile


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def entries(root):
    root = root.resolve(strict=True)
    manifest = root / "MANIFEST_SHA256.txt"
    result = {}
    for line in manifest.read_text(encoding="utf-8-sig").splitlines():
        digest, rel = line.split("  ", 1)
        pure = PurePosixPath(rel)
        path = root / rel
        if (pure.is_absolute() or ".." in pure.parts or "\\" in rel or rel in result
                or path.is_symlink() or not path.resolve().is_relative_to(root)):
            raise ValueError("Invalid manifest path: " + rel)
        if not path.is_file() or sha(path) != digest.lower():
            raise ValueError("Hash mismatch or missing file: " + str(path))
        result[rel] = digest.lower()
    if not result:
        raise ValueError("Empty manifest: " + str(manifest))
    result["MANIFEST_SHA256.txt"] = sha(manifest)
    ignored = {".git", ".venv", ".reproduced", ".pytest_cache", "__pycache__"}
    for path in root.rglob("*"):
        rel = path.relative_to(root)
        if any(part in ignored for part in rel.parts) or path.suffix == ".pyc":
            continue
        if path.is_file() and rel.as_posix() not in result:
            raise ValueError("File absent from manifest: " + rel.as_posix())
    print("MANIFEST_PASS", root.name, len(result) - 1)
    return result


def build(root, output, prefix):
    files = entries(root)
    temporary = output.with_suffix(output.suffix + ".partial")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        for rel in sorted(files):
            info = zipfile.ZipInfo(prefix + "/" + rel, date_time=(2026, 9, 20, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            with (root / rel).open("rb") as src, z.open(info, "w", force_zip64=True) as dst:
                h = hashlib.sha256()
                for block in iter(lambda: src.read(4 * 1024 * 1024), b""):
                    h.update(block)
                    dst.write(block)
                if h.hexdigest() != files[rel]:
                    raise ValueError("File changed during ZIP creation: " + rel)
    with zipfile.ZipFile(temporary) as z:
        if z.testzip() is not None:
            raise ValueError("ZIP CRC verification failed")
    if output.exists():
        if sha(output) != sha(temporary):
            temporary.unlink()
            raise ValueError("Existing archive differs; select a new output directory: " + str(output))
        temporary.unlink()
    else:
        temporary.replace(output)
    output.with_suffix(output.suffix + ".sha256").write_text(sha(output) + "  " + output.name + "\n", encoding="ascii")
    print("ARCHIVE_READY", output.name, sha(output))


def check_index(root):
    files = entries(root)
    tracked = subprocess.check_output(["git", "-C", str(root), "ls-files", "-z"]).decode("utf-8").split("\0")
    if set(filter(None, tracked)) != set(files):
        raise ValueError("Git index file list differs from the approved release manifest")
    for rel, digest in files.items():
        blob = subprocess.check_output(["git", "-C", str(root), "show", ":" + rel])
        if hashlib.sha256(blob).hexdigest() != digest:
            raise ValueError("Git index changed bytes, possibly LF/CRLF conversion: " + rel)
    print("GIT_INDEX_BYTES_PASS", len(files))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--release-root", required=True, type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--check-index", action="store_true")
    args = p.parse_args()
    root = args.release_root.resolve(strict=True)
    if args.check_index:
        check_index(root / "repository")
        return
    out = args.output_dir.resolve() if args.output_dir else root / "publish-files"
    if any(out == root / name or out.is_relative_to(root / name) for name in ("repository", "research-assets")):
        raise ValueError("Archive output must stay outside repository and research-assets")
    out.mkdir(parents=True, exist_ok=True)
    build(root / "repository", out / "AURA-KSP_repository_v1.0.0.zip", "repository")
    build(root / "research-assets", out / "AURA-KSP_research-assets_v1.0.0.zip", "research-assets")
    names = ["AURA-KSP_repository_v1.0.0.zip", "AURA-KSP_repository_v1.0.0.zip.sha256",
             "AURA-KSP_research-assets_v1.0.0.zip", "AURA-KSP_research-assets_v1.0.0.zip.sha256"]
    (out / "LOCAL_FILES.json").write_text(json.dumps({"files": [{"name": n, "sha256": sha(out/n), "bytes": (out/n).stat().st_size} for n in names]}, indent=2) + "\n")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc)) from None
