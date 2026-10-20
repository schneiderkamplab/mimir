"""Saved final-answer rescoring; no generation or reasoning substitution."""
from pathlib import Path
import json
import torch

REVISION='c23d21b0620b635a76227c604d44e43a9f0ee389'

def score_saved(input_path, output_path, encoder_path, *, language, batch_size=8):
    from bert_score import BERTScorer
    # Require a materialized revision; bert-score does not expose revision selection.
    identity=json.loads((Path(encoder_path)/'mimir_encoder_revision.json').read_text())
    if identity!={'model':'xlm-roberta-large','revision':REVISION}:raise ValueError('Unqualified BERTScore encoder')
    scorer=BERTScorer(model_type=str(encoder_path),num_layers=17,lang=language,
                      device='cpu',idf=False,rescale_with_baseline=False)
    scorer._model.float()
    rows=[json.loads(s) for s in Path(input_path).read_text().splitlines()]
    with Path(output_path).open('x') as out:
        for start in range(0,len(rows),batch_size):
            batch=rows[start:start+batch_size];valid=[i for i,r in enumerate(batch) if r['prediction'].strip()]
            values=[0.]*len(batch)
            if valid:
                with torch.no_grad():_,_,f1=scorer.score([batch[i]['prediction'] for i in valid],[batch[i]['reference'] for i in valid],batch_size=batch_size)
                for i,value in zip(valid,f1.tolist()):values[i]=value
            for row,value in zip(batch,values):
                lengths={k:len(scorer._tokenizer.encode(row[k],truncation=False)) for k in ('prediction','reference')}
                record=row|{'scores':{'bertscore_f1':value},'encoder':identity,
                            'truncated':{k:n>scorer._tokenizer.model_max_length for k,n in lengths.items()}}
                out.write(json.dumps(record,ensure_ascii=False)+'\n')
