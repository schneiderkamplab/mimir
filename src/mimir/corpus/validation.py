"""Read-only validation for the sampled token/index format; MTC IDs are reserved."""
import numpy as np


def validate_arrays(tokens, indices):
    arrays=[indices.inst_start,indices.inst_len,indices.resp_start,indices.resp_len]
    if tokens.ndim!=1 or tokens.dtype.kind not in 'iu':raise ValueError('tokens.npy must be an integer vector')
    if any(a.ndim!=1 or a.dtype.kind not in 'iu' or len(a)!=len(arrays[0]) for a in arrays):
        raise ValueError('Index arrays must be aligned integer vectors')
    # Bound workspaces even for very large index files.
    for offset in range(0,len(arrays[0]),1_000_000):
        start,plen,resp,rlen=[a[offset:offset+1_000_000] for a in arrays]
        if any(np.any(a<0) for a in (start,plen,resp,rlen)) or np.any(rlen<1):raise ValueError('Negative offsets or empty responses')
        if np.any(start>len(tokens)) or np.any(resp>len(tokens)):
            raise ValueError('Index starts outside token stream')
        if np.any(plen>len(tokens)-start) or np.any(rlen>len(tokens)-resp):
            raise ValueError('Index ends outside token stream')
