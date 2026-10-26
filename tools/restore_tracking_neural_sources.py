"""Restore exact public source/weight bytes for the isolated offline diagnostic.

Run this copied helper INSIDE a detached f58d532 worktree. It never rewrites
production gates or manufactures historical receipts. Private head/best files
are user-supplied local inputs; no private cloud identifiers are embedded.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parents[1]
REVISION='f58d532bebd8700c07fc3f529b18bd646bd14a73'
BASE_SHA='f7d127abe2b8e37b1322a19115343286a6560700c6e02fc6080b4e2426a01086'
BASE_SIZE=1_608_850_339
BASE_URL='https://media.githubusercontent.com/media/facebookresearch/gotrack/68f76055755f2a4a8967e13ece834f975f008bdf/gotrack_checkpoint.pt'
RESTORE={
    'bench/runtime/adaptation-sources.json':'44e4c8a84d602a6fbe623622a3488db014aaddf562b06748e3a13afc88693d33',
    'bench/quality_gotrack.py':'0b18516894690de0fbf3565d5d542ced17865c161e646369fa4f7e5412a28c29',
    'bench/quality_runner.py':'bca306d1342dce2566c786fed94a7564ae090e11bf42243da2ea7bc9ce841ad1'}
ARCHIVES=(
    ('facebookresearch/gotrack','68f76055755f2a4a8967e13ece834f975f008bdf','.cache/model-quality/sources/gotrack','7c7edec1820a8872892ff30957a1fa322fcdbec396da4616d46e38702f9dd02e'),
    ('thodan/bop_toolkit','fa9bc5c3a1c7fc092fc0694e69270d275278032e','.cache/model-quality/sources/gotrack/external/bop_toolkit','f11f854db50ec9ec7d702c0f10fd71082c840f20ca4083b50c54d53b4d5babf0'),
    ('facebookresearch/dinov2','12592a9bdfea0a6e4ee4767cd5a8d3195c0e3509','.cache/model-quality/sources/gotrack/external/dinov2','db796f1c3a8a1a3d4e78b26e076140d54b985ac0f263a66426955f5e03066070'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def reserve(additional=0):
    if shutil.disk_usage(ROOT).free<10*1024**3+additional:raise ValueError('must preserve10GiB free')


def copy_verified(source,target,size,digest):
    source,target=Path(source),ROOT/target
    if source.is_symlink() or source.stat().st_size!=size or sha(source)!=digest:raise ValueError('supplied artifact differs from pinned bytes')
    if target.exists():
        if target.stat().st_size!=size or sha(target)!=digest:raise ValueError('existing target differs')
        return
    reserve(size);target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_suffix(target.suffix+'.part')
    if temporary.exists():raise ValueError('unfinished artifact needs inspection')
    shutil.copyfile(source,temporary)
    if sha(temporary)!=digest:raise ValueError('copied artifact differs')
    temporary.rename(target)


def restore_sources():
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=REVISION:
        raise ValueError('run only in the isolated detached f58d532 worktree')
    git_dir=subprocess.check_output(['git','rev-parse','--git-dir'],cwd=ROOT,text=True).strip()
    common_dir=subprocess.check_output(['git','rev-parse','--git-common-dir'],cwd=ROOT,text=True).strip()
    if (ROOT/git_dir).resolve()==(ROOT/common_dir).resolve():
        raise ValueError('refusing the primary checkout; create a linked isolated worktree')
    for name,expected in RESTORE.items():
        path=ROOT/name;raw=path.read_bytes().replace(b'\r\n',b'\n').replace(b'\n',b'\r\n')
        if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('source restoration does not match exact reviewed bytes:'+name)
        path.write_bytes(raw)
    raw=subprocess.check_output(['git','diff','eddcd4386b1459f1ecdf8713ee8d047055c9cc1a',REVISION,'--','bench/quality_gotrack.py','bench/quality_runner.py'],cwd=ROOT)
    diff_expected='30520de45084ce1f948929e38331eb3b4691be9a731d15381d487c54ed262398'
    if hashlib.sha256(raw).hexdigest()!=diff_expected:raise ValueError('historical source diff differs')
    diff=ROOT/'.cache/workbench-historical-source-replacement.diff';diff.parent.mkdir(parents=True,exist_ok=True);diff.write_bytes(raw)
    for repo,revision,relative,expected in ARCHIVES:
        reserve(256*1024**2)
        raw=bytearray()
        with urllib.request.urlopen(f'https://codeload.github.com/{repo}/zip/{revision}',timeout=60) as response:
            while block:=response.read(1024**2):
                raw.extend(block)
                if len(raw)>64*1024**2:raise ValueError('source archive64MiB bound')
        if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('pinned public source archive differs')
        target=ROOT/relative;expanded=0
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            for entry in archive.infolist():
                part=Path(*Path(entry.filename).parts[1:]);destination=target/part
                if not destination.resolve().is_relative_to(target.resolve()):raise ValueError('unsafe archive path')
                if entry.is_dir():destination.mkdir(parents=True,exist_ok=True);continue
                expanded+=entry.file_size
                if expanded>128*1024**2 or entry.file_size>64*1024**2:raise ValueError('source expansion bound')
                destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(archive.read(entry))
    lock=json.loads((ROOT/'bench/runtime/adaptation-sources.json').read_bytes())
    reviewed={'bench/quality_gotrack.py','bench/quality_runner.py'}
    for name,expected in lock['gotrack']['files'].items():
        if name not in reviewed and sha(ROOT/name)!=expected:raise ValueError('restored source closure differs:'+name)
    return {'files':len(lock['gotrack']['files']),'reviewedReplacements':sorted(reviewed),'reviewDiffSHA256':diff_expected}


def public_base():
    target=ROOT/'.cache/model-quality/checkpoints/gotrack_checkpoint.pt'
    if target.exists():
        if target.stat().st_size!=BASE_SIZE or sha(target)!=BASE_SHA:raise ValueError('existing base differs')
        return
    reserve(BASE_SIZE);target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_suffix('.pt.part')
    if temporary.exists():raise ValueError('unfinished public base needs inspection')
    count=0;h=hashlib.sha256()
    with urllib.request.urlopen(BASE_URL,timeout=120) as response,temporary.open('xb') as output:
        if int(response.headers['Content-Length'])!=BASE_SIZE:raise ValueError('public base size header differs')
        while block:=response.read(4*1024**2):
            count+=len(block)
            if count>BASE_SIZE:raise ValueError('public base exceeds pinned size')
            output.write(block);h.update(block)
    if count!=BASE_SIZE or h.hexdigest()!=BASE_SHA:raise ValueError('public base bytes/hash differ')
    temporary.rename(target)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--head',type=Path);p.add_argument('--best',type=Path)
    group=p.add_mutually_exclusive_group();group.add_argument('--base',type=Path);group.add_argument('--download-public-base',action='store_true')
    args=p.parse_args();report=restore_sources()
    model='.cache/video-workbench/models/flow-terminal-20261021/'
    if args.head:copy_verified(args.head,model+'head.pt',80_165_526,'27e06a513713232f4ae542d50b8364a3bb319669451448e694bcd335a3fb9311')
    if args.best:copy_verified(args.best,model+'best.pt',80_203_390,'0c211531ffdee325e6504411f9dee07c83cc242cd8d048c115548aed6de2fa3b')
    if args.base:copy_verified(args.base,'.cache/model-quality/checkpoints/gotrack_checkpoint.pt',BASE_SIZE,BASE_SHA)
    if args.download_public_base:public_base()
    print(json.dumps(report))


if __name__=='__main__':main()
