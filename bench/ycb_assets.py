"""Small, attributed subset of original YCB scanned object assets. No archive-wide extraction."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OBJECTS = {'025_mug': 'Mug', '006_mustard_bottle': 'Mustard bottle', '024_bowl': 'Bowl',
           '011_banana': 'Banana', '035_power_drill': 'Power drill'}
SOURCE = 'https://ycb-benchmarks.s3.amazonaws.com/index.html'
BASE = 'https://ycb-benchmarks.s3.amazonaws.com/data/google/'
MAX_ARCHIVE = 32*1024*1024
FILES = ('textured.obj', 'textured.mtl', 'texture_map.png')
SHA256 = {
    '025_mug': '4b640afc1baaa940b66948d1e59426d8da2c5a16558c7c7a118bbe28d0737de5',
    '006_mustard_bottle': '992621f78a4fa6267da71418e530cd547bd066775da0242e578ddfb561d7cb5d',
    '024_bowl': 'ac73b566d4f17c3cd5956684875c3a20dd4c8777821df62f5fee81f6b7751868',
    '011_banana': 'bad03e98066dec02ea1536ff2fa58f068efa97e9dbc89fceb4418f1ec633c881',
    '035_power_drill': '23d00071cab2161b99f21d8d317c516977e32f1f893e37d8bc4edc71a3ffe02c',
}


def download(object_id, destination):
    if object_id not in OBJECTS:
        raise ValueError('Choose an object in the pinned benchmark subset.')
    destination.mkdir(parents=True, exist_ok=True)
    archive_path = destination/f'{object_id}.tgz'
    url = BASE+object_id+'_google_16k.tgz'
    if not archive_path.exists():
        partial = archive_path.with_suffix('.part')
        try:
            with urllib.request.urlopen(url, timeout=60) as response, partial.open('wb') as target:
                total = 0
                while chunk := response.read(1024*1024):
                    total += len(chunk)
                    if total > MAX_ARCHIVE:
                        raise ValueError('Scan exceeded the bounded download size.')
                    target.write(chunk)
            partial.replace(archive_path)
        except Exception:
            partial.unlink(missing_ok=True)
            raise
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    if digest != SHA256[object_id]:
        raise ValueError('Asset differs from the pinned public scan. Review its provenance before updating the hash.')
    output = destination/object_id
    output.mkdir(exist_ok=True)
    found = set()
    with tarfile.open(archive_path, 'r:gz') as archive:
        for member in archive.getmembers():
            parts = PurePosixPath(member.name).parts
            if not member.isfile() or len(parts) < 3 or parts[-2] != 'google_16k' or parts[-1] not in FILES:
                continue
            if '..' in parts or PurePosixPath(member.name).is_absolute() or member.size > MAX_ARCHIVE:
                raise ValueError('Unsafe or oversized asset entry.')
            name = parts[-1]
            if name in found:
                raise ValueError('Duplicate asset entry.')
            found.add(name)
            with archive.extractfile(member) as source, (output/name).open('wb') as target:
                shutil.copyfileobj(source, target)
    if not {'textured.obj', 'texture_map.png'}.issubset(found):
        raise ValueError('Scan did not provide the expected geometry and texture.')
    manifest = dict(objectId=object_id, name=OBJECTS[object_id], dataset='YCB Object and Model Set',
                    source=SOURCE, download=url, archiveSHA256=digest, archiveBytes=archive_path.stat().st_size,
                    license='CC BY 4.0', authors='Berk Calli, Arjun Singh, Aaron Walsman, Siddhartha Srinivasa, Pieter Abbeel, Aaron M. Dollar',
                    variant='google_16k', files=sorted(found), geometrySource='Scanned physical object; no reconstruction performed by this project')
    (output/'source.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'{OBJECTS[object_id]}: {archive_path.stat().st_size/1024/1024:.1f} MiB, SHA256 {digest}', flush=True)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--objects', nargs='+', choices=OBJECTS, default=list(OBJECTS))
    parser.add_argument('--destination', type=Path, default=ROOT/'.cache'/'ycb')
    args = parser.parse_args()
    for object_id in args.objects:
        download(object_id, args.destination)
