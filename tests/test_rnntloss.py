import pathlib, sys; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import builtins
import types
from warprnnt_pytorch.warprnnt_pytorch import RNNTLoss


def test_rnntloss_returns_zero():
    loss = RNNTLoss(debug=True)
    result = loss('a', b=1)
    assert result == 0.0
