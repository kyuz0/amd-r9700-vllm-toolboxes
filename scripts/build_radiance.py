#!/usr/bin/env python3
"""Build current Radiance release-channel source and its rolling R9700 toolbox."""
import argparse
import json
from pathlib import Path
import subprocess
import shutil
from hashlib import sha256

from release_source import argument, checkout, floating_images

ENGINE = 'localhost/r9700-radiance:build'
TOOLBOX = 'docker.io/kyuz0/amd-r9700-vllm-toolboxes:radiance'


def run(*args):
    subprocess.run([str(value) for value in args], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path('research/radiance-build'))
    parser.add_argument('--source', type=Path, help='Upstream checkout to update from the selected release')
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--memory', default='24g')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    source = args.source.resolve() if args.source else workspace / 'source'
    resolved = checkout('magiccodingman/vllm-radiance', source, 'release')
    actual = resolved['revision']
    dockerfile = floating_images((source / 'Dockerfile').read_text())
    dockerfile = dockerfile.replace('ARG ROCM_BASE=rocm/', 'ARG ROCM_BASE=docker.io/rocm/')
    dockerfile = dockerfile.replace('ARG RELEASE_BASE=ubuntu:', 'ARG RELEASE_BASE=docker.io/library/ubuntu:')
    dockerfile += '\nLABEL org.opencontainers.image.source="https://github.com/magiccodingman/vllm-radiance" org.opencontainers.image.revision="' + actual + '" org.opencontainers.image.version="' + resolved['version'] + '"\n'
    recipe = workspace / 'Dockerfile.engine'
    recipe.write_text(dockerfile)
    bases = [argument(dockerfile, name) for name in ('ROCM_BASE', 'RELEASE_BASE')]
    for base in bases:
        run('podman', 'pull', base)
    run('podman', 'build', '--pull=never', '--network=host', '--memory=' + args.memory, '-f', recipe,
        '--build-arg', 'BUILD_JOBS=' + str(args.jobs), '-t', ENGINE, source)
    converter_source = workspace / 'converter-source'
    converter = checkout('GGZ14/vllm-mxfp4', converter_source, 'main')
    shutil.copyfile(converter_source / 'fp8_mtp.py', workspace / 'fp8_mtp.py')
    run('podman', 'build', '--pull=never', '--network=host', '--memory=8g',
        '-f', repo / 'toolboxes/Dockerfile.radiance', '--build-arg', 'ENGINE_IMAGE=' + ENGINE,
        '--build-arg', 'SOURCE_REVISION=' + actual, '--build-arg', 'SOURCE_VERSION=' + resolved['version'], '-t', TOOLBOX, workspace)
    images = json.loads(subprocess.check_output(['podman', 'image', 'inspect', ENGINE, TOOLBOX], text=True))
    receipt = {'source': resolved, 'converter_source': converter, 'converter_sha256': sha256((workspace / 'fp8_mtp.py').read_bytes()).hexdigest(), 'bases': bases, 'build_jobs': args.jobs, 'memory_limit': args.memory,
               'images': [{'id': item['Id'], 'names': item.get('RepoTags', [])} for item in images]}
    (workspace / 'BUILD-IDENTITY.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
