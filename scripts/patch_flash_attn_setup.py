#!/usr/bin/env python3
"""Keep flash-attention from reinstalling its vendored AITER checkout.

The Dockerfile builds that exact checkout as a wheel first. This patch is
intentionally strict so a changed upstream setup.py fails the image build
instead of silently installing a second, potentially different AITER stack.
"""

from pathlib import Path


path = Path("setup.py")
source = path.read_text()
old = '''        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-build-isolation", "third_party/aiter"],
            check=True,
        )
'''
new = '''        # AITER is installed from this pinned submodule by the toolbox build.
        pass
'''
if source.count(old) != 1:
    raise SystemExit("Unsupported flash-attention setup.py: AITER install block changed")
path.write_text(source.replace(old, new))
print("Patched flash-attention to reuse the pinned AITER wheel")
