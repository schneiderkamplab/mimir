"""Offline Inspect wiring fixture; never part of a comparison suite."""
from inspect_ai import Task,task
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate
from inspect_ai.scorer import exact

@task
def smoke():
    return Task(dataset=[Sample(id='a',input='Return the test response.',target='Default output from mockllm/model')],
                solver=generate(),scorer=exact())
