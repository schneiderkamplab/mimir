"""Materialize pinned fork revisions plus patches in a new directory."""
import argparse,hashlib,json,subprocess
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('destination',type=Path)
    parser.add_argument('--name',choices=['transformers','vllm','all'],default='all')
    parser.add_argument('--local-source',type=Path,help='Optional read-only git source for a single fork')
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    records=json.loads((root/'provenance/forks.json').read_text())
    for name,record in records.items():
        if args.name not in ('all',name):continue
        target=args.destination.resolve()/name
        if target.exists():raise FileExistsError(target)
        target.parent.mkdir(parents=True,exist_ok=True)
        source=str(args.local_source) if args.local_source else record['url']
        subprocess.run(['git','clone','--no-checkout',*(['--shared'] if args.local_source else []),source,str(target)],check=True)
        subprocess.run(['git','-C',str(target),'checkout','--detach',record['revision']],check=True)
        patch=root/record['patch']
        if hashlib.sha256(patch.read_bytes()).hexdigest()!=record['patch_sha256']:raise ValueError('Patch hash mismatch')
        subprocess.run(['git','-C',str(target),'apply','--check',str(patch)],check=True)
        subprocess.run(['git','-C',str(target),'apply',str(patch)],check=True)
        print(target)
if __name__=='__main__':main()
