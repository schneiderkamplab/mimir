"""Experimental launch-time NCCL shim binding; never enabled implicitly."""

import ctypes
import os


class NcclCheckpointShim:
    def __init__(self):
        if not os.environ.get('NCCL_CHECKPOINT_KVS_PATH'):
            raise RuntimeError('NCCL checkpoint shim requires NCCL_CHECKPOINT_KVS_PATH')
        self.library = ctypes.CDLL(None)
        try:
            self.prepare_call = self.library.ncclCheckpointPrepare
            self.restore_call = self.library.ncclCheckpointRestore
        except AttributeError as error:
            raise RuntimeError('NCCL checkpoint shim must be LD_PRELOADed at launch') from error
        for function in (self.prepare_call, self.restore_call):
            function.argtypes = []
            function.restype = ctypes.c_int

    def prepare(self):
        result = self.prepare_call()
        if result:
            raise RuntimeError(f'ncclCheckpointPrepare failed: NCCL status {result}')

    def restore(self):
        result = self.restore_call()
        if result:
            raise RuntimeError(f'ncclCheckpointRestore failed: NCCL status {result}')
