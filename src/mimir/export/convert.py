"""Strict native Mimir2 export through the pinned Transformers integration."""
from pathlib import Path
import json
import shutil
import uuid
import yaml
import torch
from mimir.checkpoints.contract import validate


def export(checkpoint, output, tokenizer, *, ema=True):
    checkpoint,output=Path(checkpoint),Path(output)
    manifest=validate(checkpoint)
    if output.exists():raise FileExistsError(output)
    cfg=yaml.safe_load((checkpoint/'all_config.yaml').read_text())
    meta=yaml.safe_load((checkpoint/'train_metadata.yaml').read_text())
    if cfg['arch']['name']!='mimir2':raise ValueError('Only Mimir2 exports are supported')
    if cfg['arch'].get('recurrence_mode', 'hierarchical') != 'hierarchical':
        raise ValueError('Looped recurrence export is not supported by the pinned inference forks')
    if cfg['arch'].get('h_residual_persistence') or cfg['arch'].get('h_branch_l_mixing'):
        raise ValueError('Persistent H streams and branch-specific L mixing require inference fork support')
    from transformers.models.mimir2.convert_mimir2_weights_to_hf import convert_checkpoint
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tag=manifest['tag']
    if cfg['checkpoint_format']=='sharded':
        import torch.distributed.checkpoint as dcp
        reader=dcp.FileSystemReader(checkpoint/f'fsdp2_{tag}');entries=reader.read_metadata().state_dict_metadata
        sources={}
        for key,value in entries.items():
            if not key.startswith('model.'):continue
            name=key.removeprefix('model.');slot=f'optim.state.{name}.param_ema'
            if ema and name.endswith('.weight') and slot not in entries:raise ValueError(f'Missing EMA: {name}')
            sources[name]=slot if ema and slot in entries else key
        tensors={key:torch.empty(entries[key].size,dtype=entries[key].properties.dtype) for key in sources.values()}
        dcp.load(tensors,storage_reader=reader,no_dist=True)
        state={name:tensors[key] for name,key in sources.items()}
    else:
        data=torch.load(checkpoint/f'unsharded_{tag}.pt',map_location='cpu',weights_only=False);state=data['model']
        if ema:
            slots=data.get('optim',{}).get('state',{})
            if any(not isinstance(k,str) for k in slots):raise ValueError('EMA export requires named optimizer state')
            for name in state:
                slot=slots.get(name,{})
                if 'param_ema' in slot:state[name]=slot['param_ema']
                elif name.endswith('.weight'):raise ValueError(f'Missing EMA: {name}')
    native=cfg['arch'] | {k:meta[k] for k in ('vocab_size','max_seq_len')}
    model=convert_checkpoint(state,native).eval()
    tok=AutoTokenizer.from_pretrained(tokenizer,local_files_only=True)
    if not tok.chat_template:raise ValueError('Tokenizer must own an explicit chat template')
    for target in (model.config,model.generation_config):
        for key in ('bos_token_id','eos_token_id','pad_token_id'):setattr(target,key,getattr(tok,key))
    template=tok.apply_chat_template([{'role':'user','content':'Test'}],tokenize=True,add_generation_prompt=True)
    if len(tok)>model.config.vocab_size:raise ValueError('Tokenizer exceeds model vocabulary')
    stage=output.with_name('.'+output.name+'.'+uuid.uuid4().hex+'.partial');stage.mkdir(parents=True)
    try:
        model.save_pretrained(stage);tok.save_pretrained(stage)
        reloaded=AutoTokenizer.from_pretrained(stage,local_files_only=True)
        if reloaded.apply_chat_template([{'role':'user','content':'Test'}],tokenize=True,add_generation_prompt=True)!=template:
            raise ValueError('Chat template changed during serialization')
        restored=AutoModelForCausalLM.from_pretrained(stage,local_files_only=True).eval()
        ids=torch.tensor([[1,2,3]])
        with torch.no_grad():torch.testing.assert_close(model(ids).logits,restored(ids).logits,rtol=0,atol=0)
        (stage/'export_receipt.json').write_text(json.dumps({'schema':1,'architecture':'mimir2','checkpoint_manifest':manifest,'ema':ema},indent=2))
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage);raise
