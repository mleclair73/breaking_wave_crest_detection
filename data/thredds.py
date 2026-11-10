"""Fetch files from the CHL THREDDS file server and verify them against a SHA256SUMS list.

Each data folder keeps a ``SHA256SUMS`` file (``sha256sum`` format) naming the files it needs
and a ``download.py`` that maps each name to its THREDDS path. Files already present with the
right hash are skipped; a present file with the wrong hash is reported and left untouched.
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

FILESERVER = "https://chldata.erdc.dren.mil/thredds/fileServer/frf/"


def sha256(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_sums(path):
    sums = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            sums[name.lstrip("*")] = digest
    return sums


def fetch(url, dest):
    tmp = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "breaking-wave-crest-detection"})
    with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    return tmp


def sync(folder, remote_path, sums_file=None):
    """Download missing files listed in ``folder/SHA256SUMS`` and verify all of them.

    ``remote_path(name)`` returns the path under ``frf/`` on the file server.
    Returns the number of files that failed; exits nonzero from ``main`` if any did.
    """
    folder = Path(folder)
    sums = read_sums(sums_file or folder / "SHA256SUMS")
    failed = 0
    for name, digest in sums.items():
        dest = folder / name
        if dest.exists():
            ok = sha256(dest) == digest
            print(f"{'ok      ' if ok else 'MISMATCH'} {name}")
            failed += not ok
            continue
        url = FILESERVER + remote_path(name)
        try:
            tmp = fetch(url, dest)
        except OSError as e:
            print(f"FAILED   {name}: {e}")
            failed += 1
            continue
        if sha256(tmp) != digest:
            print(f"MISMATCH {name} (downloaded copy kept as {tmp.name})")
            failed += 1
            continue
        tmp.rename(dest)
        print(f"fetched  {name}")
    return failed


def main(folder, remote_path):
    failed = sync(folder, remote_path)
    if failed:
        sys.exit(f"{failed} file(s) failed verification")
