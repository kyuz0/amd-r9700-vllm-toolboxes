import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from release_source import checkout, floating_images

class ReleaseSourceTests(unittest.TestCase):
    def select(self, response, expected):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "VERSION").write_text("0.13.0")
            with patch("release_source.urllib.request.urlopen", **response), patch("release_source.subprocess.run") as run, patch("release_source.subprocess.check_output", side_effect=["", "abc123"]):
                receipt = checkout("owner/engine", root, "release")
            self.assertEqual(receipt["ref"], expected)
            self.assertIn(["git", "-C", str(root), "fetch", "origin", expected], [call.args[0] for call in run.call_args_list])
            self.assertEqual(receipt["revision"], "abc123")
    def test_latest_release_is_resolved(self):
        self.select({"return_value": io.BytesIO(json.dumps({"tag_name": "v2.0.0"}).encode())}, "v2.0.0")
    def test_only_missing_release_uses_maintainer_channel(self):
        self.select({"side_effect": HTTPError("url", 404, "missing", {}, None)}, "release")
    def test_network_failure_does_not_silently_fall_back(self):
        with patch("release_source.urllib.request.urlopen", side_effect=HTTPError("url", 503, "unavailable", {}, None)):
            with self.assertRaises(HTTPError):
                checkout("owner/engine", Path("unused"), "release")
    def test_removes_image_digest_but_keeps_compatible_source_arguments(self):
        recipe = "ARG BASE_DIGEST=" + "a"*64 + "\nARG R4D_PIN=compatible\nFROM ubuntu:24.04@sha256:" + "b"*64 + "\nFROM ${BASE_REPO}:${BASE_TAG}@sha256:${BASE_DIGEST}\n"
        result = floating_images(recipe)
        self.assertNotIn("@sha256", result)
        self.assertNotIn("BASE_DIGEST=", result)
        self.assertIn("ARG R4D_PIN=compatible", result)

if __name__ == "__main__":
    unittest.main()
