import pathlib, sys; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import types
import typeguard
if not hasattr(typeguard, "check_argument_types"):
    def check_argument_types(*a, **k):
        return True
    def check_return_type(*a, **k):
        return True
    typeguard.check_argument_types = check_argument_types
    typeguard.check_return_type = check_return_type
espnet2 = types.ModuleType("espnet2")
espnet2_asr = types.ModuleType("espnet2.asr")
decoder_mod = types.ModuleType("espnet2.asr.decoder")
abs_mod = types.ModuleType("espnet2.asr.decoder.abs_decoder")
import torch as _t
class AbsDecoder(_t.nn.Module):
    pass
abs_mod.AbsDecoder = AbsDecoder
transducer_mod = types.ModuleType("espnet2.asr.transducer")
beam_mod = types.ModuleType("espnet2.asr.transducer.beam_search_transducer")
class Hypothesis:
    def __init__(self, score=0.0, yseq=None, dec_state=None, lm_state=None):
        self.score = score
        self.yseq = yseq or []
        self.dec_state = dec_state
        self.lm_state = lm_state
class ExtendedHypothesis(Hypothesis):
    pass
beam_mod.Hypothesis = Hypothesis
beam_mod.ExtendedHypothesis = ExtendedHypothesis
transducer_mod.beam_search_transducer = beam_mod
espnet2_asr.decoder = decoder_mod
decoder_mod.abs_decoder = abs_mod
espnet2.asr = espnet2_asr
sys.modules["espnet2"] = espnet2
sys.modules["espnet2.asr"] = espnet2_asr
sys.modules["espnet2.asr.decoder"] = decoder_mod
sys.modules["espnet2.asr.decoder.abs_decoder"] = abs_mod
sys.modules["espnet2.asr.transducer"] = transducer_mod
sys.modules["espnet2.asr.transducer.beam_search_transducer"] = beam_mod
import torch
from dataclasses import dataclass

from fujielab.asr_parallel_cbs.espnet_ext.espnet2.asr.transducer.transducer_decoder import TransducerDecoder


@dataclass
class SimpleHyp:
    yseq: list
    dec_state: tuple
    score: float = 0.0


def test_init_state_shapes():
    dec = TransducerDecoder(vocab_size=10, rnn_type="lstm", num_layers=2, hidden_size=8)
    state = dec.init_state(3)
    assert state[0].shape == (2, 3, 8)
    assert state[1].shape == (2, 3, 8)


def test_select_state_and_create_batch_states():
    dec = TransducerDecoder(vocab_size=5, rnn_type="lstm", num_layers=2, hidden_size=8)
    state = dec.init_state(2)
    selected = dec.select_state(state, 1)
    assert selected[0].shape == (2, 1, 8)
    assert selected[1].shape == (2, 1, 8)

    batch = dec.create_batch_states([selected, selected])
    assert batch[0].shape == (2, 2, 8)
    assert batch[1].shape == (2, 2, 8)


def test_score_caches():
    dec = TransducerDecoder(vocab_size=5, rnn_type="lstm", num_layers=1, hidden_size=4)
    hyp = SimpleHyp(yseq=[1], dec_state=dec.init_state(1))
    cache = {}
    out, new_state, label = dec.score(hyp, cache)
    assert out.shape == (4,)
    assert label.shape == (1,)
    assert "1" in cache
