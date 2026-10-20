"""Atomic optional W&B publication, with explicit fractional-epoch axes."""

def publish_record(run,record,*,step,epoch):
    if type(step) is not int or step<0:raise ValueError('Invalid step')
    if not isinstance(epoch,(int,float)) or epoch<0:raise ValueError('Invalid epoch')
    metrics=record['metrics']
    run.define_metric('eval/step');run.define_metric('eval/epoch')
    for name in metrics:run.define_metric(name,step_metric='eval/epoch')
    run.log(dict(metrics)|{'eval/step':step,'eval/epoch':float(epoch)},commit=True)
