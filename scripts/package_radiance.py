#!/usr/bin/env python3
"""Package a tested local Radiance runtime without recompiling its compiler stack."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess

from release_source import checkout


def run(*args):
    subprocess.run([str(value) for value in args], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True, help='Already-built local Radiance image')
    parser.add_argument('--workspace', type=Path, default=Path('research/radiance-build'))
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    run('podman', 'image', 'exists', args.image)
    parent = json.loads(subprocess.check_output(['podman', 'image', 'inspect', args.image], text=True))[0]
    labels = parent.get('Labels') or {}
    assert labels.get('org.opencontainers.image.source') == 'https://github.com/magiccodingman/vllm-radiance'
    revision = labels['org.opencontainers.image.revision']
    version = labels['org.opencontainers.image.version']
    current = checkout('magiccodingman/vllm-radiance', workspace / 'source', 'release')
    run('git', '-C', workspace / 'source', 'fetch', 'origin', revision)
    changes = subprocess.check_output(['git', '-C', str(workspace / 'source'), 'diff', '--name-only',
                                       revision, current['revision']], text=True).splitlines()
    assert all(name == 'README.md' for name in changes), 'Current upstream runtime differs; review before publication'
    assert version == current['version'], 'Runtime version differs from current upstream'
    converter = checkout('GGZ14/vllm-mxfp4', workspace / 'converter-source', 'main')
    shutil.copyfile(workspace / 'converter-source/fp8_mtp.py', workspace / 'fp8_mtp.py')
    recipe = workspace / 'Dockerfile.engine'
    recipe.write_text((repo / 'toolboxes/Dockerfile.radiance').read_text())
    tag = 'docker.io/kyuz0/amd-r9700-vllm-toolboxes:radiance'
    run('podman', 'build', '--pull=never', '--network=host', '--memory=8g', '-f', recipe,
        '--build-arg', 'ENGINE_IMAGE=' + parent['Id'], '--build-arg', 'SOURCE_REVISION=' + revision,
        '--build-arg', 'SOURCE_VERSION=' + version, '-t', tag, workspace)
    image = json.loads(subprocess.check_output(['podman', 'image', 'inspect', tag], text=True))[0]
    receipt = {'source': {'repository': 'magiccodingman/vllm-radiance', 'revision': revision,
                          'version': version, 'channel': 'tested-local-build'},
               'resolved_upstream': current, 'upstream_documentation_changes': changes,
               'runtime_equivalent_to_current_upstream': True,
               'publication_basis': 'locally-built-and-qualified-runtime',
               'parent_image_id': parent['Id'], 'converter_source': converter,
               'converter_sha256': sha256((workspace / 'fp8_mtp.py').read_bytes()).hexdigest(),
               'images': [{'id': image['Id'], 'names': image.get('RepoTags', [])}]}
    (workspace / 'BUILD-IDENTITY.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
