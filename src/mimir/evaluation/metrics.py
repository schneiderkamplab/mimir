"""Aggregate sample/pass observations, never unweighted shard means."""
from pathlib import Path
import json
import os
import tempfile


def merge_samples(paths):
    seen=set(); totals={}; counts={}; rows=[]
    for path in paths:
        for line in Path(path).read_text().splitlines():
            row=json.loads(line)
            key=(row['task'],row['sample_id'],row['pass'])
            if key in seen:raise ValueError(f'Duplicate observation: {key}')
            seen.add(key);rows.append(row)
            for metric,value in row['scores'].items():
                if not isinstance(value,(int,float)) or not __import__('math').isfinite(value):
                    raise ValueError('Finite numeric sample scores required')
                k=f"{row['task']}/{metric}";totals[k]=totals.get(k,0)+value;counts[k]=counts.get(k,0)+1
    if not rows:raise ValueError('No observations')
    return {'schema':1,'observations':len(rows),'metrics':{k:totals[k]/counts[k] for k in totals},
            'sums':totals,'counts':counts}


def write_record(path, record):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix='.'+path.name,dir=path.parent)
    try:
        with os.fdopen(fd,'w') as f:
            json.dump(record,f,allow_nan=False);f.flush();os.fsync(f.fileno())
        os.replace(temp,path)
    finally:
        if os.path.exists(temp):os.unlink(temp)


def classification_metrics(pairs):
    """Macro F1 and multiclass MCC from the complete confusion matrix."""
    import math
    from collections import Counter
    pairs=list(pairs)
    if not pairs:raise ValueError('No class observations')
    labels=set(x for pair in pairs for x in pair)
    actual=Counter(a for a,p in pairs);predicted=Counter(p for a,p in pairs)
    correct=Counter(a for a,p in pairs if a==p);total=len(pairs)
    f1=sum(2*correct[c]/(actual[c]+predicted[c]) for c in labels)/len(labels)
    denominator=math.sqrt((total*total-sum(v*v for v in actual.values()))*(total*total-sum(v*v for v in predicted.values())))
    numerator=sum(correct.values())*total-sum(actual[c]*predicted[c] for c in labels)
    return {'accuracy':sum(correct.values())/total,'macro_f1':f1,'mcc':numerator/denominator if denominator else 0.}


def pass_at_k(successes,attempts,k):
    import math
    if not 0<=successes<=attempts or not 1<=k<=attempts:raise ValueError('Invalid pass statistics')
    return 1.-math.comb(attempts-successes,k)/math.comb(attempts,k) if attempts-successes>=k else 1.


def complete_task_set(paths,expected,*,classification_tasks=(),pass_k=None):
    """Require declared samples/passes before publishing any task-set aggregate."""
    from collections import defaultdict
    rows=[json.loads(line) for path in paths for line in Path(path).read_text().splitlines()]
    groups=defaultdict(list)
    for row in rows:groups[row['task']].append(row)
    if set(groups)!=set(expected):raise ValueError('Incomplete or unexpected task set')
    for task,items in groups.items():
        passes=defaultdict(set)
        for item in items:
            if item['sample_id'] in passes[item['pass']]:raise ValueError('Duplicate sample/pass')
            passes[item['pass']].add(item['sample_id'])
        wanted=expected[task]
        if len(passes)!=wanted['passes'] or any(len(ids)!=wanted['samples'] for ids in passes.values()):
            raise ValueError(f'Incomplete sample/pass coverage: {task}')
        reference=next(iter(passes.values()))
        if any(ids!=reference for ids in passes.values()):raise ValueError('Passes contain different samples')
    record=merge_samples(paths)
    for task in classification_tasks:
        by_pass=defaultdict(list)
        for row in groups[task]:by_pass[row['pass']].append((row['target_class'],row['predicted_class']))
        per_pass={str(k):classification_metrics(v) for k,v in by_pass.items()}
        record.setdefault('pass_metrics',{})[task]=per_pass
        values={key:sum(metrics[key] for metrics in per_pass.values())/len(per_pass)
                for key in ('accuracy','macro_f1','mcc')}
        record['metrics'].update({f'{task}/{k}':v for k,v in values.items()})
    for task,k in (pass_k or {}).items():
        samples=defaultdict(list)
        for row in groups[task]:samples[row['sample_id']].append(bool(row['success']))
        record['metrics'][f'{task}/pass@{k}']=sum(pass_at_k(sum(v),len(v),k) for v in samples.values())/len(samples)
    record['expected']=expected
    return record
