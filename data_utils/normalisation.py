import json
import torch


class SimpleNormalizer:
    """
    Normalizes columns: [x, r0_mm, Z].
    - x: keep as-is (already in [0,1]) or map to [-1,1] if desired
    - r0_mm: standardize or min-max to [-1,1]
    - Z: apply optional power transform (p in {1.0, 2/3, 4/3}), then standardize or min-max
    """

    def __init__(self, x_mode="identity", r0_mode="minmax", Z_mode="standard", Z_power=1.0):
        """
        x_mode: "identity" or "minmax01" or "minmax11"
        r0_mode: "standard" or "minmax11"
        Z_mode: "standard" or "minmax11"
        Z_power: 1.0 leaves Z as-is; use 2/3 to reflect TF-like scaling if needed
        """
        self.cfg = dict(x_mode=x_mode, r0_mode=r0_mode,
                        Z_mode=Z_mode, Z_power=Z_power)
        self.params = {}

    def fit(self, X):
        """
        X: [N,3] with columns [x, r0_mm, Z]
        """
        x = X[:, 0:1]
        r0 = X[:, 1:2]
        Z = X[:, 2:3]

        # x stats for minmax if needed
        if self.cfg["x_mode"] == "minmax01":
            self.params["x_min"] = float(x.min())
            self.params["x_max"] = float(x.max())
        elif self.cfg["x_mode"] == "minmax11":
            self.params["x_min"] = float(x.min())
            self.params["x_max"] = float(x.max())

        # r0
        if self.cfg["r0_mode"] == "standard":
            mu = float(r0.mean())
            sd = float(r0.std(unbiased=False) + 1e-12)
            self.params.update(r0_mu=mu, r0_sd=sd)
        elif self.cfg["r0_mode"] == "minmax11":
            self.params.update(r0_min=float(r0.min()), r0_max=float(r0.max()))
        else:
            raise ValueError("Unknown r0_mode")

        # Z (with optional power)
        Zt = Z ** self.cfg["Z_power"]
        if self.cfg["Z_mode"] == "standard":
            mu = float(Zt.mean())
            sd = float(Zt.std(unbiased=False) + 1e-12)
            self.params.update(Z_mu=mu, Z_sd=sd)
        elif self.cfg["Z_mode"] == "minmax11":
            self.params.update(Z_min=float(Zt.min()), Z_max=float(Zt.max()))
        else:
            raise ValueError("Unknown Z_mode")

    def transform(self, X):
        x = X[:, 0:1]
        r0 = X[:, 1:2]
        Z = X[:, 2:3]

        # x
        xm = self.cfg["x_mode"]
        if xm == "identity":
            x_n = x
        elif xm == "minmax01":
            denom = (self.params["x_max"] - self.params["x_min"] + 1e-12)
            x_n = (x - self.params["x_min"]) / denom
        elif xm == "minmax11":
            denom = (self.params["x_max"] - self.params["x_min"] + 1e-12)
            x_n = 2.0 * (x - self.params["x_min"]) / denom - 1.0
        else:
            raise ValueError("Unknown x_mode")

        # r0
        if self.cfg["r0_mode"] == "standard":
            r0_n = (r0 - self.params["r0_mu"]) / self.params["r0_sd"]
        else:
            denom = (self.params["r0_max"] - self.params["r0_min"] + 1e-12)
            r0_n = 2.0 * (r0 - self.params["r0_min"]) / denom - 1.0

        # Z
        Zt = Z ** self.cfg["Z_power"]
        if self.cfg["Z_mode"] == "standard":
            Z_n = (Zt - self.params["Z_mu"]) / self.params["Z_sd"]
        else:
            denom = (self.params["Z_max"] - self.params["Z_min"] + 1e-12)
            Z_n = 2.0 * (Zt - self.params["Z_min"]) / denom - 1.0

        return torch.cat([x_n, r0_n, Z_n], dim=1)

    def inverse(self, Xn):
        xn = Xn[:, 0:1]
        r0n = Xn[:, 1:2]
        Zn = Xn[:, 2:3]

        # x
        xm = self.cfg["x_mode"]
        if xm == "identity":
            x = xn
        elif xm == "minmax01":
            denom = (self.params["x_max"] - self.params["x_min"] + 1e-12)
            x = xn * denom + self.params["x_min"]
        elif xm == "minmax11":
            denom = (self.params["x_max"] - self.params["x_min"] + 1e-12)
            x = (xn + 1.0) * 0.5 * denom + self.params["x_min"]
        else:
            raise ValueError("Unknown x_mode")

        # r0
        if self.cfg["r0_mode"] == "standard":
            r0 = r0n * self.params["r0_sd"] + self.params["r0_mu"]
        else:
            denom = (self.params["r0_max"] - self.params["r0_min"] + 1e-12)
            r0 = (r0n + 1.0) * 0.5 * denom + self.params["r0_min"]

        # Z
        if self.cfg["Z_mode"] == "standard":
            Zt = Zn * self.params["Z_sd"] + self.params["Z_mu"]
        else:
            denom = (self.params["Z_max"] - self.params["Z_min"] + 1e-12)
            Zt = (Zn + 1.0) * 0.5 * denom + self.params["Z_min"]
        if self.cfg["Z_power"] != 1.0:
            Zt = torch.clamp(Zt, min=1e-12)
            Z = Zt ** (1.0 / self.cfg["Z_power"])
        else:
            Z = Zt

        return torch.cat([x, r0, Z], dim=1)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(dict(cfg=self.cfg, params=self.params), f)

    def load(self, path):
        with open(path, "r") as f:
            obj = json.load(f)
        self.cfg = obj["cfg"]
        self.params = obj["params"]


def fit_and_transform(X, cfg=None):
    """
    Convenience: fit stats on X and return normalized X plus the normalizer
    """
    cfg = cfg or dict(x_mode="identity", r0_mode="minmax11",
                      Z_mode="standard", Z_power=1.0)
    norm = SimpleNormalizer(**cfg)
    norm.fit(X)
    Xn = norm.transform(X)
    return Xn, norm
