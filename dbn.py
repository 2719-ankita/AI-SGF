
from dataclasses import dataclass
from typing import List
from sklearn.neural_network import BernoulliRBM
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

@dataclass
class DBNConfig:
    rbm_layers: List[int]
    rbm_learning_rate: float = 0.01
    rbm_n_iter: int = 20
    classifier_C: float = 1.0

def build_dbn(cfg: DBNConfig, n_features: int) -> Pipeline:
    layers = []
    in_units = n_features
    for units in cfg.rbm_layers:
        layers.append((f"rbm_{units}", BernoulliRBM(n_components=units, learning_rate=cfg.rbm_learning_rate, n_iter=cfg.rbm_n_iter, verbose=False)))
        in_units = units
    clf = LogisticRegression(C=cfg.classifier_C, max_iter=1000)
    layers.append(("logreg", clf))
    return Pipeline(layers)
