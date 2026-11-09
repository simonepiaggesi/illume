"""A small end-to-end check of the installed public API."""

import numpy as np
import torch
from importlib.metadata import version

import illume
from illume import ILLUME
from illume import core


def test_fit_transform_inverse_on_cpu(monkeypatch):
    monkeypatch.setattr(core, "device", torch.device("cpu"))

    rng = np.random.default_rng(42)
    x_train = rng.normal(size=(8, 3)).astype(np.float32)
    x_val = rng.normal(size=(4, 3)).astype(np.float32)
    model = ILLUME(latent_dim=2, max_epochs=1, batch_size=4, early_stopping=2)
    params = {
        "num_k": None,
        "l_rec": 0.0,
        "l_kld": 1.0,
        "l_so": 0.0,
        "l_co": 0.0,
        "l_st": 0.0,
    }

    train_losses, val_losses = model.fit(
        X=(x_train, x_val),
        idx_num_cat=[[0], [1], [2]],
        params_dict=params,
        seed=42,
    )
    weights, z, decoder_weights = model.transform(
        x_val, return_decoder_weights=True
    )
    reconstructed = model.inverse_transform(z, decoder_weights)

    assert illume.__version__ == version("mimosa-illume")
    assert len(train_losses) == len(val_losses) == 1
    assert np.isfinite(train_losses).all()
    assert np.isfinite(val_losses).all()
    assert weights.shape == (4, 2, 3)
    assert z.shape == (4, 2)
    assert np.isfinite(z).all()
    assert reconstructed.shape == x_val.shape
    assert np.isfinite(reconstructed).all()
