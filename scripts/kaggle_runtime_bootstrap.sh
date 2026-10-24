#!/usr/bin/env bash
# Explicit Kaggle-only installation into a fresh directory. No model acquisition.
set -euo pipefail
if [[ $# != 5 ]]; then
  echo 'Usage: bash kaggle_runtime_bootstrap.sh LAUNCHER SOURCE_INSTALL RUNTIME_ROOT UV_WHEEL UV_WHEEL_SHA256' >&2
  exit 2
fi
if [[ "$(uname -s)" != Linux || -z "${KAGGLE_KERNEL_RUN_TYPE:-}" || ! -d /kaggle/input || ! -d /kaggle/working ]]; then
  echo 'Runtime bootstrap requires the actual Kaggle Linux notebook session.' >&2
  exit 2
fi
launcher=$1
source_install=$2
runtime_root=$3
uv_wheel=$4
uv_wheel_sha256=$5
host_python=$(command -v python3)
# Notebook Python is used only for stdlib inspection and a targeted wheel install.
# No package is installed into its environment and it is never the training Python.
while IFS= read -r variable; do unset "$variable"; done < <(compgen -v | grep -E '^(PYTHON|PIP_|UV_)' || true)
unset VIRTUAL_ENV CONDA_PREFIX LD_PRELOAD LD_LIBRARY_PATH
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PIP_NO_CACHE_DIR=1
"$host_python" -I -B "$launcher" verify-source --destination "$source_install"
"$host_python" -I -B - "$launcher" "$source_install" "$runtime_root" "$uv_wheel" "$uv_wheel_sha256" <<'PY'
import hashlib,pathlib,re,sys,zipfile
launcher,source,dest,wheel=map(pathlib.Path,sys.argv[1:5]); expected=sys.argv[5]
for p in (launcher,source,dest,wheel):
 if not p.is_absolute(): raise SystemExit('All paths must be absolute')
 for part in (p,*p.parents):
  if part.is_symlink(): raise SystemExit('Symlink path component: '+str(part))
if not dest.is_relative_to(pathlib.Path('/kaggle/working')) or dest.exists() or not dest.parent.is_dir():
 raise SystemExit('Runtime root must be fresh under /kaggle/working, with existing parent')
if source.is_relative_to(dest) or dest.is_relative_to(source): raise SystemExit('Runtime and source roots must be separate')
if not re.fullmatch('[0-9a-f]{64}',expected) or hashlib.sha256(wheel.read_bytes()).hexdigest()!=expected:
 raise SystemExit('Explicit uv wheel SHA-256 mismatch')
with zipfile.ZipFile(wheel) as z:
 names=[n for n in z.namelist() if n.endswith('.dist-info/METADATA')]
 if len(names)!=1: raise SystemExit('Ambiguous uv wheel metadata')
 metadata=z.read(names[0]).decode()
 if '\nName: uv\n' not in '\n'+metadata or '\nVersion: 0.12.22\n' not in '\n'+metadata:
  raise SystemExit('Requires an authentic uv==0.12.22 wheel; provide its independently verified hash')
dest.mkdir()
PY
export UV_PYTHON_INSTALL_DIR="$runtime_root/python"
export UV_CACHE_DIR="$runtime_root/uv-cache"
export UV_NO_PROGRESS=1 UV_NO_TELEMETRY=1
"$host_python" -I -B -m pip install --isolated --disable-pip-version-check --no-deps --no-cache-dir --target "$runtime_root/uv-tool" "$uv_wheel"
uv_bin="$runtime_root/uv-tool/uv/uv"
[[ "$("$uv_bin" --version)" == 'uv 0.12.22' ]] || { echo 'Pinned uv binary version mismatch' >&2; exit 2; }
"$uv_bin" python install 3.11.13
managed_python=$("$uv_bin" python find --managed-python 3.11.13)
"$host_python" -I -B - "$managed_python" "$UV_PYTHON_INSTALL_DIR" <<'PY'
import pathlib,subprocess,sys
p,root=map(pathlib.Path,sys.argv[1:])
if not p.resolve().is_relative_to(root.resolve()):raise SystemExit('Managed Python escaped isolated install directory')
if subprocess.check_output([str(p),'-I','-c','import platform;print(platform.python_version())'],text=True).strip()!='3.11.13':
 raise SystemExit('Exact Python patch mismatch')
PY
"$uv_bin" venv --python "$managed_python" "$runtime_root/venv"
training_python="$runtime_root/venv/bin/python"
"$uv_bin" pip install --python "$training_python" 'pip==24.3.1' 'setuptools==75.3.0' 'wheel==0.44.0'
"$training_python" -I -B -m pip install --isolated --no-cache-dir 'torch==2.5.1+cu124' 'torchvision==0.20.1+cu124' --index-url https://download.pytorch.org/whl/cu124
printf '%s\n' 'torch==2.5.1+cu124' 'torchvision==0.20.1+cu124' > "$runtime_root/torch-constraints.txt"
"$training_python" -I -B -m pip install --isolated --no-cache-dir -c "$runtime_root/torch-constraints.txt" -r "$source_install/visualizeit-cloud-v18/bench/runtime/requirements-ranker.txt" --index-url https://pypi.org/simple
"$training_python" -I -B -m pip check
"$training_python" -I -B -m pip freeze --all > "$runtime_root/resolved-requirements.txt"
"$host_python" -I -B "$launcher" runtime-probe --python "$training_python" > "$runtime_root/runtime-observed.json"
"$host_python" -I -B - "$runtime_root" "$source_install/visualizeit-cloud-v18" "$uv_wheel" "$uv_wheel_sha256" "$managed_python" <<'PY'
import hashlib,json,os,pathlib,sys
root,source,wheel=map(pathlib.Path,sys.argv[1:4])
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def descriptor(p):return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
runtime=json.loads((root/'runtime-observed.json').read_text())
if runtime['python']!='3.11.13' or runtime['packages'].get('torch')!='2.5.1+cu124' or runtime['packages'].get('torchvision')!='0.20.1+cu124':
 raise SystemExit('Resolved runtime differs from pinned contract; installation remains incomplete')
links=[]
for parent,dirs,files in os.walk(root,followlinks=False):
 for name in dirs+files:
  p=pathlib.Path(parent)/name
  if p.is_symlink():
   target=os.readlink(p)
   lexical=pathlib.Path(os.path.abspath(p.parent/target))
   resolved=p.resolve(strict=True)
   if not lexical.is_relative_to(root) or not resolved.is_relative_to(root) or (resolved.is_dir() and p.parent.is_relative_to(resolved)):
    raise SystemExit('Runtime link escapes or cycles: '+str(p))
   links.append({'path':p.relative_to(root).as_posix(),'target':target})
receipt={'schema':'kaggle_isolated_runtime_v1','uv_version':'0.12.22','uv_wheel':descriptor(wheel),
 'runtime_root':str(root),'runtime_links':sorted(links,key=lambda x:x['path']),
 'python_request':'3.11.13','managed_python':sys.argv[5],
 'python_install_provenance':'https://github.com/astral-sh/uv/blob/0.12.22/docs/concepts/python-versions.md',
 'runtime':runtime,'resolved_requirements':descriptor(root/'resolved-requirements.txt'),
 'requirements':{p:sha(source/p) for p in ('bench/runtime/requirements-ranker.txt','bench/runtime/requirements-quality.txt')},
 'assets_acquired':False,'training_executed':False}
with (root/'runtime-receipt.json').open('x') as f:json.dump(receipt,f,indent=2,sort_keys=True);f.write('\n')
PY
echo "Isolated runtime receipt: $runtime_root/runtime-receipt.json"
echo 'Model assets and CUDA/parity/resource admission remain separate prerequisites.'
