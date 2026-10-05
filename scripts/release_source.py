"""Resolve an upstream release or its documented maintenance channel at build time."""
import json
from pathlib import Path
import re
import subprocess
import urllib.error
import urllib.request


def checkout(repository, directory, fallback):
    request = urllib.request.Request(f"https://api.github.com/repos/{repository}/releases/latest", headers={"Accept": "application/vnd.github+json", "User-Agent": "r9700-toolbox-build"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            release = json.load(response)
        ref, channel = release["tag_name"], "stable-release"
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
        ref, channel = fallback, "maintainer-channel"
    directory = Path(directory)
    if directory.exists():
        dirty = subprocess.check_output(["git", "-C", str(directory), "status", "--porcelain"], text=True).strip()
        if dirty:
            raise SystemExit("Refusing to replace a dirty upstream checkout")
        subprocess.run(["git", "-C", str(directory), "fetch", "origin", ref], check=True)
    else:
        subprocess.run(["git", "clone", "--no-checkout", f"https://github.com/{repository}.git", str(directory)], check=True)
        subprocess.run(["git", "-C", str(directory), "fetch", "origin", ref], check=True)
    subprocess.run(["git", "-C", str(directory), "checkout", "--detach", "FETCH_HEAD"], check=True)
    revision = subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
    return {"repository": repository, "ref": ref, "channel": channel, "revision": revision, "version": (directory / "VERSION").read_text().strip()}


def floating_images(recipe):
    # Follow the selected upstream's compatibility stack, with mutable image tags.
    recipe = re.sub(r"@sha256:(?:[0-9a-f]{64}|\$\{BASE_DIGEST\})", "", recipe)
    recipe = re.sub(r"^ARG BASE_DIGEST=.*\n", "", recipe, flags=re.M)
    return recipe


def argument(recipe, name):
    match = re.search(r"^ARG " + re.escape(name) + r"=(.+)$", recipe, re.M)
    if not match:
        raise ValueError(f"Selected upstream recipe no longer declares {name}")
    return match.group(1).strip()
