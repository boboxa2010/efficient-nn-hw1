import numpy as np
from scipy.optimize import least_squares
from sklearn.linear_model import LinearRegression


def fit_latency(data):
    f = data["flops"].to_numpy() / 1e9
    d = data["bytes_moved"].to_numpy() / 1e9
    t = data["latency"].to_numpy()

    def error(params):
        t0, a, c = params
        predicted = t0 + np.maximum(a * f, c * d)
        return (predicted - t) / t

    fits = []
    for scale in [0.1, 1, 10]:
        start = [t.min() / 2, np.median(t / f) * scale, np.median(t / d)]
        fit = least_squares(
            error,
            start,
            bounds=([0, 1e-15, 1e-15], [np.inf, np.inf, np.inf]),
            x_scale="jac",
        )
        fits.append(fit)

    best = min(fits, key=lambda fit: np.sum(fit.fun**2))
    t0, a, c = best.x

    return float(t0), float(1e9 / a), float(1e9 / c)


def fit_energy(data):
    x = np.column_stack(
        [
            data["flops"].to_numpy() / 1e9,
            data["bytes_moved"].to_numpy() / 1e9,
            data["latency_energy"].to_numpy(),
        ]
    )
    e = data["energy"].to_numpy()

    scale = np.linalg.norm(x, axis=0)

    model = LinearRegression(positive=True, fit_intercept=False)
    model.fit(x / scale, e, sample_weight=1 / e**2)

    a, c, p_base = model.coef_ / scale
    return float(a / 1e9), float(c / 1e9), float(p_base)
