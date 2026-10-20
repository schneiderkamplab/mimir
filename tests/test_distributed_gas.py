"""Two-rank Gloo parity for token-weighted GAS and DDP."""
from types import SimpleNamespace
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from mimir.training.state import TrainState
from mimir.training.steps import train_accumulated_batches
from mimir.model.adam_atan2 import AdamATan2

class Loss(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.tensor([.3,.7]))
    def forward(self,*,batch,carry,**kwargs):
        per=((batch['x']@self.weight)-batch['y']).square()
        loss=per.mean();count=loss.new_tensor(len(per))
        return None,loss,{'loss':(per.sum().detach(),count)}

def batch(x,y):
    return {'inputs':torch.zeros(len(y),dtype=torch.int64),'labels':torch.zeros(len(y),dtype=torch.int64),
            'x':torch.tensor(x),'y':torch.tensor(y)}

def worker(rank,rendezvous,output):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=f'file://{rendezvous}',rank=rank,world_size=2)
    try:
        a=Loss();wrapped=DDP(a);b=Loss()
        def state(m):return TrainState(m,None,AdamATan2(m.parameters(),lr=.01,ema=.9),1,10,torch.float32,False)
        cfg=SimpleNamespace(compile_train_batch_mode='default',fsdp_accumulation_sync_mode='no_sync',fsdp_shard_degree=None,
            gradient_clip_norm=None,gradient_skip_norm=None,experiment_update_probe_interval=0,resume_trace=False)
        sa,sb=state(wrapped),state(b)
        # Equal token counts per rank; unequal microbatch sizes within each rank.
        x=torch.tensor([[1.,2.],[3.,4.],[5.,6.]])+rank;y=torch.tensor([1.,0.,2.])
        micro=[batch(x[:1].tolist(),y[:1].tolist()),batch(x[1:].tolist(),y[1:].tolist())]
        train_accumulated_batches(cfg,rank,sa,micro,use_compiled=False)
        full_x=torch.cat([torch.tensor([[1.,2.],[3.,4.],[5.,6.]])+r for r in range(2)])
        loss=((full_x@b.weight)-y.repeat(2)).square().mean();loss.backward();sb.optim.step()
        torch.testing.assert_close(a.weight,b.weight,rtol=1e-6,atol=1e-7)
        for key,value in sa.optim.state[a.weight].items():
            torch.testing.assert_close(value,sb.optim.state[b.weight][key],rtol=1e-6,atol=1e-7)
    finally:dist.destroy_process_group()

def test_ddp_gas_matches_global_batch(tmp_path):
    mp.spawn(worker,args=(str(tmp_path/'rendezvous'),str(tmp_path)),nprocs=2,join=True)
