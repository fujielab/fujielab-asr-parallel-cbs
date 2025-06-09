import pytest
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import types
espnet2 = types.ModuleType('espnet2')
espnet2_asr_transducer = types.ModuleType('espnet2.asr_transducer')
espnet2_decoder = types.ModuleType('espnet2.asr.decoder')
espnet2_abs = types.ModuleType('espnet2.asr.decoder.abs_decoder')
class AbsDecoder:
    pass
espnet2_abs.AbsDecoder = AbsDecoder
sys.modules.setdefault('espnet2', espnet2)
sys.modules.setdefault('espnet2.asr_transducer', espnet2_asr_transducer)
sys.modules.setdefault('espnet2.asr.decoder', espnet2_decoder)
sys.modules.setdefault('espnet2.asr.decoder.abs_decoder', espnet2_abs)
jn_mod = types.ModuleType('espnet2.asr_transducer.joint_network')
jn_mod.JointNetwork = lambda *a, **k: None
sys.modules.setdefault('espnet2.asr_transducer.joint_network', jn_mod)

import torch

from fujielab.asr_parallel_cbs.espnet_ext.espnet.nets.beam_search_parallel_transducer_online import (
    BeamSearchParallelTransducer,
    Hypothesis,
)

class DummyDecoder:
    def __init__(self, vocab_size=10, blank_id=0):
        self.odim = vocab_size
        self.blank_id = blank_id
        self.device = torch.device('cpu')
    def set_device(self, device):
        self.device = device

class DummyJoint:
    pass

class DummyLM:
    rnn_type = 'dummy'


def make_search(lm=False):
    dec = DummyDecoder()
    joint = DummyJoint()
    lm_obj = DummyLM() if lm else None
    return BeamSearchParallelTransducer(decoder=dec, joint_network=joint, beam_size=2, lm=lm_obj)


def test_recombine_hyps():
    search = make_search()
    hyps = [
        Hypothesis(score=-1.0, yseq=[0, 1]),
        Hypothesis(score=-2.0, yseq=[0, 1]),
        Hypothesis(score=-0.5, yseq=[0]),
    ]
    out = search.recombine_hyps(hyps)
    assert len(out) == 2
    for h in out:
        if h.yseq == [0, 1]:
            import numpy as np
            assert h.score == pytest.approx(np.logaddexp(-1.0, -2.0))


def test_store_restore():
    search = make_search()
    search.search_cache = [Hypothesis(score=0.0, yseq=[0])]
    search.store()
    search.search_cache.append(Hypothesis(score=-1.0, yseq=[1]))
    search.restore()
    assert len(search.search_cache) == 1
    assert search.search_cache[0].yseq == [0]


def test_create_lm_batch_inputs_padding():
    search = make_search(lm=True)
    seqs = [[0, 1, 2], [0, 3]]
    batch = search.create_lm_batch_inputs(seqs)
    sos = search.sos
    assert batch.tolist() == [[sos, 1, 2], [sos, 0, 3]]
