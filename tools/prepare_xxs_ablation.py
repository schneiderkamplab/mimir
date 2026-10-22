"""Prepare held native Mimir jobs. This module never executes campaign jobs."""
import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import torch
from omegaconf import OmegaConf

from mimir.corpus.sampled import V1DatasetIndices
from mimir.corpus.validation import validate_arrays
from mimir.model.mimir2 import Mimir2
from mimir.model.lm_head import LMHead
from mimir.scheduling.store import Store
from mimir.training.entry import configuration

REPO = Path(__file__).resolve().parents[1]
RECIPE = REPO / 'campaigns/xxs-dfm15-20k/recipe.json'


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inspect_data(path):
    """Check headers and bounded index windows; do not iterate the training set."""
    path = Path(path).resolve()
    metadata = json.loads((path/'metadata.json').read_text())
    tokens = np.load(path/'tokens.npy', mmap_mode='r')
    indices = {f.name: np.load(path/'epoch_0'/f'{f.name}.npy', mmap_mode='r') for f in fields(V1DatasetIndices)}
    rows = len(indices['inst_start'])
    if not rows or any(a.ndim != 1 or len(a) != rows for a in indices.values()):
        raise ValueError('Misaligned or empty corpus index arrays')
    for offset in sorted({0, rows//2, max(0, rows-1024)}):
        validate_arrays(tokens, V1DatasetIndices(**{k: v[offset:offset+1024] for k,v in indices.items()}))
    receipt = {'path': str(path), 'metadata_sha256': digest(path/'metadata.json'),
               'metadata': metadata, 'epoch_0_rows': rows, 'stored_tokens': len(tokens),
               'validation_scope': 'headers and up to 3072 index rows; no token-content scan or training iteration',
               'arrays': {k: {'shape': list(v.shape), 'dtype': str(v.dtype)} for k,v in indices.items()}}
    for name in ('ready.json', 'validated.json'):
        if (path/name).exists():
            receipt[name] = {'sha256': digest(path/name), 'content': json.loads((path/name).read_text())}
    return receipt


def count_parameters(arch):
    # Meta tensors allocate no model storage and never initialize a GPU.
    with torch.device('meta'):
        model = LMHead(Mimir2(arch), arch)
    excluded = sum(p.numel() for n,p in model.named_parameters() if n.startswith(('embed_tokens.', 'lm_head.')))
    total = sum(p.numel() for p in model.parameters())
    return {'capacity': total-excluded, 'token_embedding_and_lm_head': excluded, 'total': total}


def prepare(output, data, tokenizer, template, *, gpus=None, eval_steps=None, database=None):
    output, data, tokenizer, template = map(lambda p: Path(p).absolute(), (output,data,tokenizer,template))
    if output.exists():
        raise FileExistsError('Use a fresh preparation directory')
    recipe = json.loads(RECIPE.read_text())
    eval_steps = sorted(set(eval_steps or recipe['default_eval_steps']))
    if not eval_steps or eval_steps[-1] != recipe['steps'] or any(s < 1 or s > recipe['steps'] or s%5000 for s in eval_steps):
        raise ValueError('Evaluation steps must be 5K boundaries including final 20K')
    gpu_ids = [] if gpus is None else gpus.split(',')
    if gpus is not None and (not gpu_ids or any(not x.isdigit() for x in gpu_ids) or len(set(gpu_ids)) != len(gpu_ids)):
        raise ValueError('GPU assignment must be distinct numeric indices')
    data_info = inspect_data(data)
    meta = data_info['metadata']
    if meta['total_length'] < recipe['steps']*196608:
        raise ValueError('Insufficient tokens in one epoch for the fixed training budget')
    output.mkdir(parents=True)
    # Only tokenizer data and the dataset-owned template are copied, never weights/code.
    tok_out = output/'tokenizer'; tok_out.mkdir()
    for name in ('tokenizer.json','tokenizer_config.json','special_tokens_map.json','added_tokens.json'):
        if (tokenizer/name).exists(): shutil.copyfile(tokenizer/name,tok_out/name)
    shutil.copyfile(template,tok_out/'chat_template.jinja')
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_out,local_files_only=True)
    if len(tok) > meta['tokenizer_info']['vocab_size']:
        raise ValueError('Tokenizer exceeds corpus vocabulary')
    messages = [{'role':'user','content':'Test'}]
    if tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True) != tok.apply_chat_template(
            messages,tokenize=False,add_generation_prompt=True,enable_thinking=False):
        raise ValueError('Export-owned template must default to the dataset non-thinking generation policy')
    data_info['tokenizer'] = {'source': str(tokenizer), 'tokenizer_sha256': digest(tok_out/'tokenizer.json'),
                             'template_source': str(template), 'template_sha256': digest(template), 'vocab_size': len(tok)}
    write(output/'data.json',data_info)
    py = sys.executable  # Do not resolve the venv interpreter symlink away.
    resources = [f'gpu:{g}' for g in gpu_ids] or ['gpu:UNASSIGNED']
    env = {'CUDA_VISIBLE_DEVICES': ','.join(gpu_ids), 'MIMIR_WANDB':'0',
           'HF_HOME': str(output/'cache/huggingface')}
    jobs, counts, previous_evals = [], {}, []
    def add(identity, kind, config, dependencies=()):
        jobs.append({'id':identity,'kind':kind,'config':config,'held':True,'max_attempts':1 if kind=='train' else 3,
                     'depends':[{'id':d,'policy':p} for d,p in dependencies]})
    for variant, extra in recipe['variants'].items():
        run = output/'runs'/variant
        checkpoints = run/'checkpoints'
        common = recipe['common_overrides'] + extra + [f'data.path={data}', f'checkpoint_path={checkpoints}', f'run_name={recipe["id"]}-{variant}']
        cfg = configuration(common)
        arch = OmegaConf.to_container(cfg.arch,resolve=True) | {'vocab_size':meta['tokenizer_info']['vocab_size'],
                       'max_seq_len':meta['max_seq_len']-1, 'fwd_bwd_dtype':cfg.fwd_bwd_dtype}
        counts[variant] = count_parameters(arch)
        (output/'configs').mkdir(exist_ok=True)
        OmegaConf.save(cfg,output/'configs'/f'{variant}.yaml')
        write(output/'configs'/f'{variant}.overrides.json',common)
        prior_step, last_train = 0, None
        current_evals = []
        for step in eval_steps:
            identity = f'{variant}.step_{step}'
            train_id = identity+'.train'
            overrides = common + [f'stop_after_step={step}']
            dependencies = [(x,'terminal') for x in (current_evals if prior_step else previous_evals)]
            if prior_step:
                overrides += [f'resume_checkpoint_path={checkpoints}', f'resume_checkpoint_tag=step_{prior_step}']
                dependencies += [(last_train,'success')]
            command = [py,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={len(gpu_ids) or 1}',
                       '--module','mimir','train',*overrides]
            add(train_id,'train',{'command':command,'cwd':str(REPO),'env':env,'resources':resources,
                'progress_file':str(checkpoints/'progress.json')},dependencies)
            checkpoint_id=identity+'.checkpoint'
            path=checkpoints/f'step_{step}'
            add(checkpoint_id,'checkpoint_wait',{'path':str(path),'timeout':60},[(train_id,'success')])
            export_id=identity+'.export'; exported=run/'exports'/f'step_{step}'
            add(export_id,'export',{'command':[py,'-m','mimir','export',str(path),str(exported),'--tokenizer',str(tok_out)],
                                  'cwd':str(REPO),'env':env|{'CUDA_VISIBLE_DEVICES':''}},[(checkpoint_id,'success')])
            current_evals=[]
            eval_index=0
            for suite_name in recipe['suites']:
                source=REPO/'src/mimir/evaluation/suites'/f'{suite_name}.json'
                protocol=json.loads(source.read_text())
                inputs, expected, suite_evals = [], {}, []
                for task in protocol['tasks']:
                    task_id=task['id']; eval_id=identity+'.'+suite_name+'.'+task_id
                    suite_path=output/'suites'/f'{suite_name}.{task_id}.json'
                    write(suite_path, {'schema':1,'id':suite_name+'.'+task_id,'source_sha256':digest(source),'tasks':[task]})
                    result=run/'eval'/f'step_{step}'/suite_name/task_id
                    eval_gpu = gpu_ids[eval_index % len(gpu_ids)] if gpu_ids else ''
                    eval_index += 1
                    add(eval_id,'eval',{'evaluation':{'suite':str(suite_path),'model':str(exported),'output':str(result)},
                        'server':{'model':str(exported),'gpus':eval_gpu,'port':8100+int(eval_gpu or 0),'eager':False,'max_model_len':meta['max_seq_len']-1},
                        'controls':{'max_connections':4},'resources':[f'gpu:{eval_gpu}'] if eval_gpu else resources},[(export_id,'success')])
                    current_evals.append(eval_id);suite_evals.append(eval_id)
                    if task['backend']=='inspect':
                        inputs.append(str(result/f'{task_id}.samples.jsonl'))
                        expected[task.get('aggregate_task',task_id)]={'samples':len(task.get('sample_ids',[])) or task['samples'], 'passes':task['passes']}
                if inputs:
                    # Keep per-task metric records. Do not invent a scalar mean
                    # across different benchmarks or average nonlinear scores.
                    add(identity+'.'+suite_name+'.merge','merge',{'inputs':inputs,'expected':expected,
                        'output':str(run/'metrics'/f'step_{step}.{suite_name}.json')},[(x,'success') for x in suite_evals])
            prior_step,last_train=step,train_id
        previous_evals=current_evals
    base=counts['default']['capacity']
    for name,c in counts.items():
        c['ratio_to_default']=c['capacity']/base
        c['difference_from_double_percent']=(c['capacity']/(2*base)-1)*100
        if name!='default' and abs(c['difference_from_double_percent']) > 2:
            raise ValueError(f'{name} is no longer within 2% of the doubling target')
    plan={'schema':1,'id':recipe['id'],'jobs':jobs}
    write(output/'plan.json',plan);write(output/'parameters.json',counts)
    if database is None:
        database=Path(tempfile.mkdtemp(prefix='mimir-xxs-dfm15-20k-'))/'queue.sqlite'
    database=Path(database).absolute()
    if database.exists():raise FileExistsError(database)
    store=Store(database)
    for job in jobs:
        store.add(job['id'],job['kind'],job['config'],[(d['id'],d['policy']) for d in job['depends']],job['max_attempts'],held=True)
    store.backup(output/'queue.backup.sqlite')
    rows=store.rows()
    assert all(r['state']=='pending' and r['held'] and r['attempt']==0 for r in rows)
    write(output/'preparation.json',{'status':'prepared_not_started','recipe_sha256':digest(RECIPE),
        'database':str(database),'jobs':len(jobs),'held_jobs':len(rows),'attempts':0,'gpu_assignment':gpu_ids,
        'eval_steps':eval_steps,'steps_per_variant':recipe['steps'],'global_batch_tokens':196608,
        'token_budget_per_variant':recipe['steps']*196608,'python':py,
        'notes':['DFM15 receipt status is preserved in data.json; no promotion performed',
                 'GPU/FA4 and full evaluation qualification required before release',
                 'Capacity-matched, not FLOP-matched; dense mixtures retain default execution',
                 'Live queue must be on local disk; queue.backup.sqlite is a snapshot only']})
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--tokenizer',type=Path,required=True)
    parser.add_argument('--template',type=Path,required=True)
    parser.add_argument('--gpus',help='Optional future assignment; even bound jobs remain held')
    parser.add_argument('--eval-steps',type=int,nargs='+')
    parser.add_argument('--database',type=Path,help='Fresh path on local disk; default is a new directory under /tmp')
    args=parser.parse_args()
    print(prepare(args.output,args.data,args.tokenizer,args.template,gpus=args.gpus,eval_steps=args.eval_steps,database=args.database))

if __name__=='__main__':main()
