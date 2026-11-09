"""quant.py - int8 quantisation of the frozen logistic-regression pipeline (imputer + scaler + logreg).
quantize(pipe, X_train): derive int8 parameters from a fitted float pipeline and its training data.
int8_logit(q, X):        integer inference (as the firmware will do it) -> logit and class.
float_logit(pipe, X):    reference float logit.
Inputs are standardised in float (cheap on the ESP32 FPU), then quantised; the dot product is integer.
"""
import numpy as np

import config as C

# int8 values are kept in the symmetric range [-127, 127].
QMAX = 2 ** (C.Q_BITS - 1) - 1          # 127


# Applies the fitted imputer (missing -> median) and scaler (z-score) of the pipeline.
def _z(pipe, X):
    imp, sc = pipe[0], pipe[1]
    return sc.transform(imp.transform(X))      # X keeps its column names (no sklearn warning)


# Reference result: the float logistic-regression score.
def float_logit(pipe, X):
    lr = pipe[-1]
    return _z(pipe, X) @ lr.coef_.ravel() + lr.intercept_[0]


# Chooses the two scales: inputs are clipped at a high percentile of |z| on the training data,
# weights at their largest magnitude; the bias is stored as an int32 in the product scale.
def quantize(pipe, X_train):
    imp, sc, lr = pipe[0], pipe[1], pipe[-1]
    z = _z(pipe, X_train)
    clip = float(np.percentile(np.abs(z), C.Q_INPUT_CLIP_PCT))
    s_x = clip / QMAX
    w = lr.coef_.ravel()
    s_w = float(np.max(np.abs(w))) / QMAX
    return {"median": imp.statistics_.copy(), "mean": sc.mean_.copy(), "scale": sc.scale_.copy(),
            "clip": clip, "s_x": s_x, "s_w": s_w,
            "qw": np.clip(np.round(w / s_w), -QMAX, QMAX).astype(np.int8),
            "qb": int(np.round(lr.intercept_[0] / (s_x * s_w)))}


# Integer inference exactly as on the ESP32: int8 inputs x int8 weights summed in int32.
def int8_logit(q, X):
    X = np.asarray(X, dtype=np.float64).copy()
    nan = np.isnan(X)
    X[nan] = np.take(q["median"], np.nonzero(nan)[1])
    z = (X - q["mean"]) / q["scale"]
    qx = np.clip(np.round(z / q["s_x"]), -QMAX, QMAX).astype(np.int32)
    acc = qx @ q["qw"].astype(np.int32) + q["qb"]               # int32 accumulator
    return acc * q["s_x"] * q["s_w"], (acc > 0).astype(int)
