#!/usr/bin/env python3
"""Build the current GGZ14 maintenance release and rolling R9700 toolboxes."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from hashlib import sha256
from release_source import argument, checkout, floating_images

ENGINE = 'localhost/r9700-ggz14:build'
TOOLBOX = 'docker.io/kyuz0/amd-r9700-vllm-toolboxes:ggz14-tp2'
SINGLE_ENGINE = 'localhost/r9700-ggz14-single:build'
SINGLE_TOOLBOX = 'docker.io/kyuz0/amd-r9700-vllm-toolboxes:ggz14-tp1'


def run(*args):
    subprocess.run([str(value) for value in args], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path('research/ggz14-build'))
    parser.add_argument('--source', type=Path, help='Upstream checkout to update from the selected release')
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--memory', default='24g')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    source = args.source.resolve() if args.source else workspace / 'source'
    resolved = checkout('GGZ14/vllm-mxfp4', source, 'main')
    actual = resolved['revision']
    dockerfile = floating_images((source / 'Dockerfile.ggz14.top').read_text())
    base = argument(dockerfile, 'BASE_REPO') + ':' + argument(dockerfile, 'BASE_TAG')
    if not base.startswith(('docker.io/', 'ghcr.io/')):
        base = 'docker.io/' + base
    r4d = argument(dockerfile, 'R4D_PIN')
    run('python3', repo / 'scripts/normalize_radiance_base.py', '--workspace', workspace, '--source', base)
    marker = 'ENV SP=/opt/vllm/lib/python3.12/site-packages\n'
    if dockerfile.count(marker) != 1:
        raise SystemExit('Upstream Dockerfile layout changed')
    dockerfile = dockerfile.replace(marker, marker + f'ENV MAX_JOBS={args.jobs}\n')
    dockerfile += '\nLABEL org.opencontainers.image.source="https://github.com/GGZ14/vllm-mxfp4" org.opencontainers.image.revision="' + actual + '" org.opencontainers.image.version="' + resolved['version'] + '"\n'
    recipe = workspace / 'Dockerfile.engine'
    recipe.write_text(dockerfile)
    run('podman', 'build', '--pull=never', '--network=host', '--memory=' + args.memory, '-f', recipe,
        '--build-arg', 'BASE_REPO=localhost/r9700-radiance-base', '--build-arg', 'BASE_TAG=build', '--build-arg', 'R4D_PIN=' + r4d,
        '--build-arg', 'GFX_ARCH=gfx1201', '-t', ENGINE, source)
    shutil.copyfile(source / 'fp8_mtp.py', workspace / 'fp8_mtp.py')
    run('podman', 'build', '--pull=never', '--network=host', '--memory=8g', '-f', repo / 'toolboxes/Dockerfile.ggz14',
        '--build-arg', 'ENGINE_IMAGE=' + ENGINE, '--build-arg', 'SOURCE_REVISION=' + actual, '--build-arg', 'SOURCE_VERSION=' + resolved['version'], '-t', TOOLBOX, workspace)
    # TP1 uses the launcher's rx9 narrow-state kernels; TP2 preserves rx6.
    shutil.copyfile(source / 'r4d_radiance_extras_rx9.patch', workspace / 'r4d_radiance_extras_rx9.patch')
    run('podman', 'build', '--pull=never', '--network=host', '--memory=16g',
        '-f', repo / 'toolboxes/Dockerfile.ggz14-single', '--build-arg', 'ENGINE_IMAGE=' + ENGINE, '--build-arg', 'BASE_IMAGE=localhost/r9700-radiance-base:build', '--build-arg', 'R4D_REF=' + r4d,
        '-t', SINGLE_ENGINE, workspace)
    run('podman', 'build', '--pull=never', '--network=host', '--memory=8g',
        '-f', repo / 'toolboxes/Dockerfile.ggz14', '--build-arg', 'ENGINE_IMAGE=' + SINGLE_ENGINE,
        '--build-arg', 'SOURCE_REVISION=' + actual, '--build-arg', 'SOURCE_VERSION=' + resolved['version'], '-t', SINGLE_TOOLBOX, workspace)
    images = json.loads(subprocess.check_output(['podman', 'image', 'inspect',
                        ENGINE, TOOLBOX, SINGLE_ENGINE, SINGLE_TOOLBOX], text=True))
    receipt = {'source': resolved, 'libr4d_revision': r4d, 'base': base,
               'converter_sha256': sha256((source / 'fp8_mtp.py').read_bytes()).hexdigest(),
               'single_gpu_patch_sha256': sha256((source / 'r4d_radiance_extras_rx9.patch').read_bytes()).hexdigest(),
               'images': [{'id': record['Id'], 'names': record.get('RepoTags', [])} for record in images]}
    (workspace / 'BUILD-IDENTITY.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
