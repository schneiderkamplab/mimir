import copy
from types import SimpleNamespace
import pytest
import torch
from torch.utils.checkpoint import checkpoint
from mimir.model.expert_mixtures import TopKRouter
from mimir.model.routing_balance import BalanceStep


def test_selection_only_bias_and_sign_update():
    router=TopKRouter(2,3,2,balance_enabled=True,balance_update_rate=.1)
    with torch.no_grad():
        for bias,value in zip(router.biases,[2.,1.,0.]):bias.fill_(value)
        router.selection_bias.copy_(torch.tensor([-10.,0.,10.]))
    x=torch.ones(4,2,requires_grad=True)
    with BalanceStep(router) as step:
        weights=router(x)
        expected=torch.tensor([0.,torch.sigmoid(torch.tensor(1.)),.5]);expected/=expected.sum()
        torch.testing.assert_close(weights,expected.expand(4,-1))
        weights[:,1].sum().backward()
        assert router.biases[1].grad.abs().sum()>0
        step.commit()
    torch.testing.assert_close(router.selection_bias,torch.tensor([-9.9,-.1,9.9]))
    assert 'selection_bias' not in dict(router.named_parameters())


@pytest.mark.parametrize('compiled',[False,True])
@pytest.mark.parametrize('checkpointed',[False,True])
def test_gas_and_checkpoint_replay_count_once(compiled,checkpointed):
    router=TopKRouter(2,3,1,balance_enabled=True)
    def forward(x):return checkpoint(router,x,use_reentrant=False) if checkpointed else router(x)
    fn=torch.compile(forward,backend='eager') if compiled else forward
    with BalanceStep(router) as step:
        for size in [2,5]:
            fn(torch.ones(size,2,requires_grad=True))[:,0].sum().backward()
        torch.testing.assert_close(step.counts[router],torch.tensor([7,0,0]))
        step.commit()
    torch.testing.assert_close(router.selection_bias,torch.tensor([-.001,.001,.001]))
    saved=copy.deepcopy(router.state_dict())
    other=TopKRouter(2,3,1,balance_enabled=True);other.load_state_dict(saved)
    torch.testing.assert_close(router(torch.ones(3,2)),other(torch.ones(3,2)))


def test_no_update_outside_training_step_and_eval():
    router=TopKRouter(2,3,1,balance_enabled=True)
    router(torch.ones(3,2))
    assert not router.selection_bias.any()
    with BalanceStep(router.eval()) as step:
        router(torch.ones(3,2));step.commit()
    assert not router.selection_bias.any()
    assert 'selection_bias' not in TopKRouter(2,3,1).state_dict()


def _distributed_worker(rank, rendezvous):
    import torch.distributed as dist
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2)
    try:
        router=TopKRouter(2,3,1,balance_enabled=True)
        with BalanceStep(router) as step:
            if rank==0:router(torch.ones(7,2))
            step.commit()
        torch.testing.assert_close(router.selection_bias,torch.tensor([-.001,.001,.001]))
    finally:dist.destroy_process_group()


def test_global_counts_include_idle_ranks(tmp_path):
    torch.multiprocessing.spawn(_distributed_worker,args=(str(tmp_path/'rdzv'),),nprocs=2,join=True)


class RoutedLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.router=TopKRouter(2,3,1,balance_enabled=True)
    def forward(self,*,batch,carry,**kwargs):
        weights=self.router(batch['x'])
        loss=(weights*weights.new_tensor([1.,2.,3.])).sum(-1).mean()
        count=loss.new_tensor(len(weights))
        return None,loss,{'loss':(loss.detach()*count,count)}


@pytest.mark.parametrize('skip,enabled',[(False,True),(True,True),(False,False)])
def test_training_step_commits_only_after_optimizer(skip,enabled):
    from mimir.training.steps import train_accumulated_batches
    from mimir.training.state import TrainState
    from mimir.model.adam_atan2 import AdamATan2
    model=RoutedLoss()
    state=TrainState(model,None,AdamATan2(model.parameters(),lr=.01,ema=.9),1,10,torch.float32,False)
    cfg=SimpleNamespace(compile_train_batch_mode='default',fsdp_accumulation_sync_mode='no_sync',
        fsdp_shard_degree=None,gradient_clip_norm=None,gradient_skip_norm=1e-8 if skip else None,
        experiment_update_probe_interval=0,resume_trace=False)
    batches=[{'inputs':torch.zeros(n,dtype=torch.long),'labels':torch.zeros(n,dtype=torch.long),
              'x':torch.ones(n,2)} for n in [2,5]]
    _,skipped=train_accumulated_batches(cfg,0,state,batches,use_compiled=False,optimizer_step_enabled=enabled)
    assert skipped == skip
    expected=torch.tensor([-.001,.001,.001]) if enabled and not skip else torch.zeros(3)
    torch.testing.assert_close(model.router.selection_bias,expected)


def test_exported_router_compiles_and_preserves_selection_bias():
    from transformers import Mimir2Config
    from transformers.models.mimir2.modeling_mimir2 import Mimir2Router
    from vllm.model_executor.models.mimir2 import Mimir2Router as ServingRouter
    native=TopKRouter(4,3,2,balance_enabled=True).eval()
    with torch.no_grad():
        native.selection_bias.copy_(torch.tensor([-.2,.1,.3]))
        for row in native.rows:row.normal_()
    exported=Mimir2Router(Mimir2Config(hidden_size=4,num_attention_heads=1),3,2,True).eval()
    exported.load_state_dict(native.state_dict())
    assert ServingRouter is Mimir2Router
    compiled=torch.compile(exported,backend='aot_eager',fullgraph=True)
    x=torch.randn(5,4)
    torch.testing.assert_close(native(x),compiled(x),rtol=0,atol=0)


def test_optimizer_and_bias_resume_match_uninterrupted():
    from mimir.model.adam_atan2 import AdamATan2
    model=RoutedLoss();optimizer=AdamATan2(model.parameters(),lr=.01,ema=.9)
    def step(model,optimizer):
        optimizer.zero_grad()
        with BalanceStep(model) as balance:
            _,loss,_=model(batch={'x':torch.ones(7,2)},carry=None)
            loss.backward();optimizer.step();balance.commit()
    step(model,optimizer)
    restored=RoutedLoss();other=AdamATan2(restored.parameters(),lr=.01,ema=.9)
    restored.load_state_dict(copy.deepcopy(model.state_dict()))
    other.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    step(model,optimizer);step(restored,other)
    for key,value in model.state_dict().items():
        torch.testing.assert_close(value,restored.state_dict()[key],rtol=0,atol=0)
    for a,b in zip(model.parameters(),restored.parameters()):
        for key,value in optimizer.state[a].items():
            torch.testing.assert_close(value,other.state[b][key],rtol=0,atol=0)


def test_bias_stays_fp32_when_parameters_change_precision():
    router=TopKRouter(4,3,1,balance_enabled=True)
    router.selection_bias.copy_(torch.tensor([.0011,-.22341,.234532]))
    expected=router.selection_bias.clone()
    router.bfloat16()
    assert router.rows[0].dtype==torch.bfloat16
    assert router.selection_bias.dtype==torch.float32
    torch.testing.assert_close(router.selection_bias,expected,rtol=0,atol=0)


def test_older_disabled_checkpoint_architecture_still_matches():
    from mimir.training.checkpoint_io import same_architecture
    previous={'name':'mimir2'}
    current=previous|{'moe_aux_loss_free':False,'mos_aux_loss_free':False,'routing_bias_update_rate':.001}
    assert same_architecture(previous,current)
    assert not same_architecture(previous,current|{'moe_aux_loss_free':True})
    assert not same_architecture(current,current|{'routing_bias_update_rate':.002})
