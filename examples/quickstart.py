"""Small end-to-end ILLUME example using generated data."""

import numpy as np

from illume import ILLUME

rng = np.random.default_rng(42)
x_train = rng.normal(size=(8, 3)).astype(np.float32)
x_val = rng.normal(size=(4, 3)).astype(np.float32)
model = ILLUME(latent_dim=2, max_epochs=1, batch_size=4)
model.fit(
    X=(x_train, x_val),
    idx_num_cat=[[0], [1], [2]],
    params_dict={
        "num_k": None,
        "l_rec": 0.0,
        "l_kld": 1.0,
        "l_so": 0.0,
        "l_co": 0.0,
        "l_st": 0.0,
    },
    seed=42,
)
weights, z, decoder_weights = model.transform(x_val, return_decoder_weights=True)
reconstructed = model.inverse_transform(z, decoder_weights)
print("latent shape:", z.shape)
print("reconstructed shape:", reconstructed.shape)
