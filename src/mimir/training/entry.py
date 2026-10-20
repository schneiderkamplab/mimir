from hydra import compose, initialize_config_module
from omegaconf import OmegaConf


def configuration(overrides):
    with initialize_config_module(config_module='mimir.conf', version_base=None):
        cfg=compose(config_name='train', overrides=overrides)
    from mimir.training.config import PretrainConfig
    PretrainConfig(**OmegaConf.to_container(cfg,resolve=True))
    return cfg


def run(overrides, suspendable=False):
    if suspendable and not any(x.startswith('suspend_control_dir=') for x in overrides):
        checkpoint=next((x.split('=',1)[1] for x in overrides if x.startswith('checkpoint_path=')),None)
        if not checkpoint:
            raise ValueError('--suspendable requires checkpoint_path=<directory>')
        from pathlib import Path
        overrides=list(overrides)+[f'suspend_control_dir={Path(checkpoint).resolve() / "control"}']
    cfg=configuration(overrides)
    if suspendable:
        import os,sys,subprocess
        if 'LOCAL_RANK' in os.environ:
            raise ValueError('Run one suspension controller outside torchrun, not one per rank')
        if cfg.suspension_backend == 'cuda':
            if not cfg.suspension_settings:
                raise ValueError('CUDA suspension requires suspension_settings=<shim/settings.json>')
            return subprocess.run([sys.executable,'-m','mimir.training.suspension.launcher',
                '--settings',cfg.suspension_settings,'--control-root',cfg.suspend_control_dir,'--',
                sys.executable,'-m','mimir','train',*overrides],check=True)
        from mimir.training.disk_recovery import supervise
        return supervise(overrides,cfg)
    from mimir.training.loop import launch
    launch(cfg)
