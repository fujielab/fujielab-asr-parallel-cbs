import pathlib, sys; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import types
esp_mod = types.ModuleType("espnet2.asr_transducer.activation")
import torch as _t
ESP_ACT=getattr(_t.nn, "Tanh")
esp_mod.get_activation=lambda *a, **k: ESP_ACT()
sys.modules.setdefault("espnet2", types.ModuleType("espnet2"))
sys.modules.setdefault("espnet2.asr_transducer", types.ModuleType("espnet2.asr_transducer"))
sys.modules["espnet2.asr_transducer.activation"] = esp_mod
import torch
from fujielab.asr_parallel_cbs.espnet_ext.espnet2.asr_transducer.joint_network import JointNetwork


def test_jointnetwork_forward_shape():
    batch, t, u = 2, 3, 4
    enc_size, dec_size, out_size = 5, 6, 7
    joint = JointNetwork(
        output_size=out_size,
        encoder_size=enc_size,
        decoder_size=dec_size,
        joint_space_size=8,
    )
    enc_out = torch.randn(batch, t, 1, enc_size)
    dec_out = torch.randn(batch, 1, u, dec_size)
    out = joint(enc_out, dec_out)
    assert out.shape == (batch, t, u, out_size)
