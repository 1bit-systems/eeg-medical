"""Pin the zuna package API contract that the training loop depends on.

Catches silent upstream signature changes and guards against the invented
APIs that previously lived in scripts/train.py.
"""

import ast
import inspect

from eeg_medical.models.zuna_loader import build_zuna11


def test_encoder_decoder_forward_accepts_train_zuna_kwargs():
    from zuna.inference.AY2l.lingua.apps.AY2latent_bci.transformer import EncoderDecoder

    params = set(inspect.signature(EncoderDecoder.forward).parameters)
    required = {
        "encoder_input", "decoder_input", "t", "chan_pos",
        "chan_pos_discrete", "chan_id", "t_coarse", "seq_lens", "target",
    }
    assert required <= params


def test_processor_process_accepts_train_zuna_kwargs():
    from zuna.inference.AY2l.lingua.apps.AY2latent_bci.eeg_data import EEGProcessor

    params = set(inspect.signature(EEGProcessor.process).parameters)
    required = {
        "eeg_signal", "chan_pos", "chan_pos_discrete", "chan_id",
        "t_coarse", "seq_lens", "max_tc", "token_dropout",
    }
    assert required <= params


def test_bcidatasetargs_imports_from_eeg_data():
    from zuna.inference.AY2l.lingua.apps.AY2latent_bci.eeg_data import (  # noqa: F401
        BCIDatasetArgs,
    )


def test_build_zuna11_kwargs_are_real_fields():
    """Every kwarg build_zuna11 passes to DecoderTransformerArgs must exist."""
    from zuna.inference.AY2l.lingua.apps.AY2latent_bci.transformer import (
        DecoderTransformerArgs,
    )

    src = inspect.getsource(build_zuna11)
    tree = ast.parse(src)
    call = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "DecoderTransformerArgs"
    )
    kwargs = {kw.arg for kw in call.keywords}
    fields = set(DecoderTransformerArgs.__dataclass_fields__)
    assert kwargs <= fields


def test_zuna_has_no_public_zuna_class():
    """load_zuna_pretrained must not depend on `zuna.ZUNA` (it never existed)."""
    import zuna

    assert not hasattr(zuna, "ZUNA")
