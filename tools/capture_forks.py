from pathlib import Path
import subprocess, hashlib, json, argparse
root=Path(__file__).resolve().parents[1]; out=root/'patches'; out.mkdir(exist_ok=True)
parser=argparse.ArgumentParser();parser.add_argument('sources',type=Path);args=parser.parse_args()
records={}
for name in ('transformers','vllm'):
 repo=args.sources/name
 def git(*args):return subprocess.check_output(['git','-C',str(repo),*args])
 tracked=git('diff','--binary','HEAD')
 additions=[]
 for rel in git('ls-files','--others','--exclude-standard').decode().splitlines():
  if '__pycache__' in rel or not ('mimir2' in rel):continue
  proc=subprocess.run(['git','diff','--no-index','--binary','--','/dev/null',rel],cwd=repo,stdout=subprocess.PIPE)
  if proc.returncode not in (0,1):raise RuntimeError(rel)
  additions.append(proc.stdout)
 patch=tracked+b''.join(additions);(out/f'{name}.patch').write_bytes(patch)
 records[name]={'url':git('remote','get-url','origin').decode().strip(),'revision':git('rev-parse','HEAD').decode().strip(),'patch':f'patches/{name}.patch','patch_sha256':hashlib.sha256(patch).hexdigest()}
(root/'provenance/forks.json').write_text(json.dumps(records,indent=2)+'\n')
