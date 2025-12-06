import numpy as np
import torch

from ml.data_generator import (
    BalancedTransactionDataset,
    balance_by_oversampling,
    build_balanced_dataloader,
    run_label_propagation,
)


def make_clusters():
    rng = np.random.default_rng(0)
    X_a = rng.normal(loc=-2.0, scale=0.3, size=(20, 4))
    X_b = rng.normal(loc=2.0, scale=0.3, size=(20, 4))
    y = np.array(["Food"] * len(X_a) + ["Misc"] * len(X_b))
    return np.vstack([X_a, X_b]), y


def test_run_label_propagation_adds_confident_samples():
    X_labeled, y_labeled = make_clusters()
    rng = np.random.default_rng(1)
    # полю неразмеченных точек вокруг центров
    X_unlabeled = np.vstack([
        rng.normal(loc=-2.1, scale=0.2, size=(10, 4)),
        rng.normal(loc=2.1, scale=0.2, size=(10, 4)),
    ])
    X_aug, y_aug, pseudo_idx, pseudo_y = run_label_propagation(
        X_labeled,
        y_labeled,
        X_unlabeled,
        confidence_threshold=0.8,
    )
    assert X_aug.shape[0] >= X_labeled.shape[0]
    assert set(np.unique(y_aug)) == {"Food", "Misc"}
    assert pseudo_idx.shape[0] == pseudo_y.shape[0]


def test_balance_by_oversampling_equalizes_counts():
    X = np.random.randn(30, 3)
    y = np.array(["Food"] * 10 + ["Misc"] * 20)
    X_bal, y_bal, indices = balance_by_oversampling(X, y, random_state=123, return_indices=True)
    unique, counts = np.unique(y_bal, return_counts=True)
    assert np.all(counts == counts[0])
    assert X_bal.shape[1] == X.shape[1]
    assert indices.shape[0] == X_bal.shape[0]


def test_build_balanced_dataloader_shapes():
    X = np.random.randn(40, 5)
    y = np.array(["Food"] * 10 + ["Misc"] * 30)
    loader, encoder = build_balanced_dataloader(
        X,
        y,
        batch_size=8,
        random_state=7,
    )
    batch = next(iter(loader))
    features, labels = batch
    assert isinstance(features, torch.Tensor)
    assert features.shape[1] == 5
    assert isinstance(labels, torch.Tensor)
    # label encoder должен знать о двух классах
    assert set(encoder.classes_) == {"Food", "Misc"}
