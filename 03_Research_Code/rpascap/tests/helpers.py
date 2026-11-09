"""Synthetic instances for unit tests and solver checks (NOT study data)."""
import numpy as np
from rpascap.model.two_stage import ConfigData, Instance


def random_instance(seed=0, K=2, configs_per_k=3, lanes=(0, 2, 3), T=4, B=2, W=3,
                    eps=False, theta_A=1e-5):
    rng = np.random.default_rng(seed)
    configs = []
    for k in range(K):
        cl = []
        for c in range(configs_per_k):
            n = lanes[c % len(lanes)]
            U = rng.uniform(20, 60, (T, W, n)) * (rng.random((T, W, n)) > 0.15)
            C = rng.uniform(0.005, 0.08, (T, W, n))
            kap = np.zeros((T, W, n, n))
            for l in range(n):
                kap[:, :, l, l] = rng.uniform(5e-8, 3e-7, (T, W))
                if l + 1 < n:
                    v = rng.uniform(1e-7, 5e-7, (T, W))
                    kap[:, :, l, l + 1] = v; kap[:, :, l + 1, l] = v
            cl.append(ConfigData(n, 0.0 if n == 0 else rng.uniform(1, 4), U, C, kap))
        configs.append(cl)
    O = rng.poisson(25, (K, T, W)).astype(float)
    P = rng.poisson(20, (K, T)).astype(float)
    block = np.repeat(np.arange(B), int(np.ceil(T / B)))[:T]
    return Instance(configs=configs, O=O, P=P, p=np.full(W, 1.0 / W), block_of_t=block,
                    tau_h=np.full(K, 0.1), theta_A=theta_A,
                    eps=np.full(T, 2e-6) if eps else None, max_switches=1)
