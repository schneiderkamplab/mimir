"""Recover allowlisted protocol fields without copying credentials or model args."""
from pathlib import Path
import zipfile,json,struct,hashlib,zstandard,ast,argparse
parser=argparse.ArgumentParser();parser.add_argument('source',type=Path);parser.add_argument('output',type=Path);args=parser.parse_args()
base=args.source
def read(z,name):
 try:return z.read(name)
 except NotImplementedError:
  info=z.getinfo(name)
  with open(z.filename,'rb') as f:
   f.seek(info.header_offset);head=f.read(30);n,e=struct.unpack_from('<HH',head,26);f.seek(n+e,1);data=f.read(info.compress_size)
  return zstandard.ZstdDecompressor().decompress(data,max_output_size=info.file_size)
rows=[]
for folder in ['logs_ie_rerun','logs_drop_rerun','logs_hellaswag_fmt_test','logs_winogrande_ie','logs_mmlu_5shot_ie','logs_gsm8k_ie','logs_math_ie','logs_humaneval_ie','runs/dfm11_XL_epoch10_step_2850000_ema_hf','runs/three_model_full_20260922','runs/ouro_full_20260925']:
 for p in sorted((base/folder).rglob('*.eval')):
  try:
   with zipfile.ZipFile(p) as z:
    name=next((n for n in z.namelist() if n=='header.json'),None)
    if not name:continue
    h=json.loads(read(z,name));ev=h.get('eval',{});cfg=ev.get('config',{});gen=h.get('plan',{}).get('config',{})
    safe_cfg={k:cfg.get(k) for k in ('limit','sample_id','sample_shuffle','epochs','epochs_reducer','max_samples') if k in cfg}
    safe_gen={k:gen.get(k) for k in ('max_tokens','temperature','top_p','stop_seqs','seed') if k in gen}
    dataset=ev.get('dataset',{});dataset={k:dataset.get(k) for k in ('name','samples','shuffled','sample_ids','location') if k in dataset and k!='location'}
    rows.append({'source':str(p.relative_to(base)),'header_sha256':hashlib.sha256(read(z,name)).hexdigest(),'status':h.get('status'),
     'task':ev.get('task'),'dataset':dataset,'selection':safe_cfg,'generation':safe_gen,
     'scorers':[s.get('name') for s in h.get('results',{}).get('scores',[])] if h.get('results') else []})
  except (ValueError,zipfile.BadZipFile,KeyError):continue
args.output.write_text(json.dumps(rows,indent=2)+'\n')
print(f'Recovered {len(rows)} allowlisted log headers')
