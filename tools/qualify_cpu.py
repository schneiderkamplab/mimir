"""Small isolated CPU campaign. Writes only under the supplied output directory."""
from pathlib import Path
import argparse,json,os,subprocess,sys
import numpy as np
import torch

def equal(a,b):
    if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b,(a,b)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);args=parser.parse_args()
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    data=root/'corpus';epoch=data/'epoch_0';epoch.mkdir(parents=True)
    rows,length=32,9
    np.save(data/'tokens.npy',np.arange(rows*length,dtype=np.uint32)%31+1)
    for n,a in {'inst_start':np.arange(rows)*length,'inst_len':np.full(rows,3),'resp_start':np.arange(rows)*length+3,'resp_len':np.full(rows,6)}.items():np.save(epoch/f'{n}.npy',a.astype(np.int64))
    (data/'metadata.json').write_text(json.dumps({'max_seq_len':length,'total_length':rows*(length-1),'tokenizer_info':{'vocab_size':256}}))
    common=['size@arch=XXS',f'data.path={data}','accelerator_type=cpu','distributed_strategy=none','checkpoint_format=unsharded','compile_train_batch=false','fwd_bwd_dtype=float32','global_batch_size=16','epochs=1','max_steps=4','arch.hidden_size=16','arch.num_heads=2','arch.n_layers=2','arch.expansion=1','lr_warmup_steps=0','log_interval=1']
    def train(name,extra):
        with (root/f'{name}.log').open('w') as log:
            subprocess.run([sys.executable,'-m','mimir','train',*common,f'checkpoint_path={root/name}',*extra],env=os.environ|{'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2'},stdout=log,stderr=subprocess.STDOUT,check=True)
    train('baseline',['stop_after_step=4'])
    train('split',['stop_after_step=2'])
    from tokenizers import Tokenizer,models,pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    t=Tokenizer(models.WordLevel({str(i):i for i in range(256)},unk_token='0'));t.pre_tokenizer=pre_tokenizers.Whitespace()
    tok=PreTrainedTokenizerFast(tokenizer_object=t,unk_token='0',bos_token='1',eos_token='2',pad_token='0');tok.chat_template="{% for message in messages %}{{ message['content'] }}{% endfor %}";tok.save_pretrained(root/'tokenizer')
    from mimir.export.convert import export
    export(root/'split/step_2',root/'export',root/'tokenizer',ema=True)
    from mimir.evaluation.clients import evaluate
    suite=Path(__file__).resolve().parents[1]/'src/mimir/evaluation/suites/cpu-smoke-v1.json'
    evaluate(str(suite),str(root/'export'),str(root/'evaluation'))
    train('resumed',[f'resume_checkpoint_path={root}/split','resume_checkpoint_tag=step_2','stop_after_step=4'])
    train('suspended',['--suspendable','suspend_after_steps=[2]','stop_after_step=4'])
    expected=torch.load(root/'baseline/step_4/unsharded_step_4.pt',weights_only=False)
    for name in ('resumed','suspended'):
        equal(expected,torch.load(root/name/'step_4/unsharded_step_4.pt',weights_only=False))
        equal(torch.load(root/'baseline/step_4/rng.0.pt',weights_only=False),
              torch.load(root/name/'step_4/rng.0.pt',weights_only=False))
        baseline=[json.loads(s) for s in (root/'baseline/metrics.jsonl').read_text().splitlines() if 'train/loss' in s]
        actual=[json.loads(s) for s in (root/name/'metrics.jsonl').read_text().splitlines() if 'train/loss' in s]
        equal(baseline[-len(actual):],actual)
    (root/'result.json').write_text(json.dumps({'status':'passed','device':'cpu','steps':4,'exact':['weights','optimizer','EMA','RNG','losses','recurrence/LR schedules','sample position'],'campaign':['train','checkpoint','EMA export','local evaluation','resume','disk suspension']},indent=2)+'\n')
    print(root/'result.json')
if __name__=='__main__':main()
