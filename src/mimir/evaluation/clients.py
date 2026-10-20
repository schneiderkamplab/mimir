"""Direct Inspect AI and EuroEval clients with explicit frozen protocol settings."""
from pathlib import Path
import hashlib
import importlib
import json
import os
from .metrics import write_record


class BackendOOM(RuntimeError):
    """Confirmed CUDA allocation failure reported by an evaluation backend."""


def evaluate(suite,model,output,base_url='http://127.0.0.1:8000/v1',controls=None):
    controls=controls or {};path=Path(suite);protocol=json.loads(path.read_text())
    if protocol.get('schema')!=1:raise ValueError('Unsupported suite schema')
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    receipt={'suite_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'model':model,'tasks':[]}
    for item in protocol['tasks']:
        if item.get('blocked'):raise ValueError(f"Protocol {item['id']} is not qualified: {item['blocked']}")
        backend=item['backend']
        if backend=='inspect':
            from inspect_ai import eval as inspect_eval
            task=item['task']
            if ':' in task:
                module,name=task.split(':',1)
                if not module.startswith('mimir.evaluation.tasks.'):raise ValueError('Unregistered task module')
                task=getattr(importlib.import_module(module),name)(**item.get('task_args',{}))
                if item.get('samples') and len(task.dataset)!=item['samples']:
                    raise ValueError(f"Dataset count drift for {item['id']}")
            logs=inspect_eval(task,model=f"{item.get('provider','openai')}/{model}",model_base_url=base_url,
                 log_dir=str(out/item['id']),epochs=item.get('passes',1),
                 limit=item.get('limit'),sample_id=item.get('sample_ids'),
                 sample_shuffle=item.get('shuffle_seed'),
                 max_connections=controls.get('max_connections',4),
                 **item.get('generation',{}))
            failed=[log for log in logs if log.status!='success']
            if failed:
                errors='\n'.join(str(log.error) for log in failed)
                if any(marker in errors for marker in ('torch.OutOfMemoryError','CUDA out of memory')):
                    raise BackendOOM(errors)
                raise RuntimeError(f"Inspect task failed: {item['id']}: {errors}")
            with (out / f"{item['id']}.samples.jsonl").open('x') as samples:
                for log in logs:
                    for sample in log.samples or []:
                        numeric={}
                        for name,score in (sample.scores or {}).items():
                            value=score.value
                            if value in ('C','I'):value=1. if value=='C' else 0.
                            if isinstance(value,(int,float)):numeric[name]=value
                            elif isinstance(value,dict):
                                numeric.update({f'{name}/{k}':v for k,v in value.items() if isinstance(v,(int,float))})
                        samples.write(json.dumps({'task':item.get('aggregate_task',item['id']),
                            'sample_id':f"{item['id']}:{sample.id}",'pass':sample.epoch,
                            'scores':numeric,'raw':sample.model_dump(mode='json')},ensure_ascii=False)+'\n')
            receipt['tasks'].append({'id':item['id'],'logs':[log.location for log in logs]})
        elif backend=='euroeval':
            from importlib.metadata import version
            if version('euroeval') != item['version']:raise ValueError('EuroEval protocol version mismatch')
            from euroeval import Benchmarker
            benchmarker=Benchmarker(api_base=base_url,api_key=os.environ.get('OPENAI_API_KEY'),
                cache_dir=str(out/item['id']),raise_errors=True,save_results=True,
                num_iterations=item['passes'],evaluate_test_split=item['split']=='test',
                few_shot=item['few_shot'])
            results=benchmarker.benchmark(model=model,dataset=item['dataset'])
            serial=[r.model_dump(mode='json') if hasattr(r,'model_dump') else r for r in results]
            write_record(out/f"{item['id']}.json",serial)
            receipt['tasks'].append({'id':item['id'],'results':serial})
        elif backend=='cpu_fixture':
            import torch
            from transformers import AutoModelForCausalLM
            native=AutoModelForCausalLM.from_pretrained(model,local_files_only=True).eval()
            rows=[]
            for i,example in enumerate(item['examples']):
                ids=torch.tensor([example['input_ids']])
                with torch.no_grad():logits=native(ids,use_cache=False).logits
                loss=torch.nn.functional.cross_entropy(logits[:,:-1].flatten(0,1),ids[:,1:].flatten()).item()
                rows.append({'task':item['id'],'sample_id':i,'pass':1,'scores':{'cross_entropy':loss}})
            (out/f"{item['id']}.jsonl").write_text(''.join(json.dumps(r)+'\n' for r in rows))
            receipt['tasks'].append({'id':item['id'],'samples':len(rows)})
        else:raise ValueError(f'Unsupported backend: {backend}')
    write_record(out/'receipt.json',receipt)
    return receipt
