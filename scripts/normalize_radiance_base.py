#!/usr/bin/env python3
"""Normalize mixed OCI media descriptors without changing runtime content."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--source", required=True, help="Mutable upstream image tag")
    parser.add_argument("--destination", default="localhost/r9700-radiance-base:build")
    args = parser.parse_args()
    root = args.workspace.resolve() / "base-normalization"
    root.mkdir(parents=True, exist_ok=True)
    remote = "docker://" + args.source
    original = subprocess.check_output(["skopeo", "inspect", "--raw", remote])
    manifest = json.loads(original)
    if manifest.get("mediaType") not in {"application/vnd.oci.image.manifest.v1+json", "application/vnd.docker.distribution.manifest.v2+json"}:
        raise SystemExit("Expected a single-platform image manifest")
    content = {"config": manifest["config"]["digest"], "layers": [item["digest"] for item in manifest["layers"]]}
    manifest["mediaType"] = "application/vnd.oci.image.manifest.v1+json"
    manifest["config"]["mediaType"] = "application/vnd.oci.image.config.v1+json"
    changes = 0
    for item in manifest["layers"]:
        if item["mediaType"] == "application/vnd.docker.image.rootfs.diff.tar.gzip":
            item["mediaType"] = "application/vnd.oci.image.layer.v1.tar+gzip"
            changes += 1
        if item["mediaType"] not in {"application/vnd.oci.image.layer.v1.tar+gzip", "application/vnd.oci.image.layer.v1.tar"}:
            raise SystemExit("Unsupported layer media type")
    encoded = json.dumps(manifest, separators=(",", ":")).encode()
    digest = "sha256:" + hashlib.sha256(encoded).hexdigest()
    inspection = subprocess.run(["podman", "image", "inspect", args.destination], capture_output=True, text=True)
    reuse = inspection.returncode == 0 and json.loads(inspection.stdout)[0].get("Digest") == digest
    if not reuse:
        subprocess.run(["skopeo", "copy", remote, "dir:" + str(root)], check=True)
        # Fail if the rolling tag moved between inspection and download.
        downloaded = json.loads((root / "manifest.json").read_bytes())
        if content != {"config": downloaded["config"]["digest"], "layers": [item["digest"] for item in downloaded["layers"]]}:
            raise SystemExit("Base tag changed during acquisition; restart the build")
        (root / "manifest.original.json").write_bytes(original)
        (root / "manifest.json").write_bytes(encoded)
        subprocess.run(["skopeo", "copy", "dir:" + str(root), "containers-storage:" + args.destination], check=True)
    receipt = {"source": args.source, "destination": args.destination, "normalized_manifest": digest, "config_and_layers": content, "changed_media_descriptors": changes, "runtime_content_changed": False, "reused": reuse}
    (args.workspace / "BASE-NORMALIZATION.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
