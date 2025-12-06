import numpy as np
import joblib

from ml.nn_model import TorchNNClassifier


def _make_dataset():
    rng = np.random.default_rng(0)
    class_a = rng.normal(loc=0.0, scale=1.0, size=(50, 4))
    class_b = rng.normal(loc=2.0, scale=1.0, size=(50, 4))
    X = np.vstack([class_a, class_b]).astype(np.float32)
    y = np.array(["Food"] * 50 + ["Misc"] * 50)
    return X, y


def test_torch_nn_classifier_fit_predict(tmp_path):
    X, y = _make_dataset()
    clf = TorchNNClassifier(
        max_epochs=5,
        batch_size=16,
        random_state=123,
        device="cpu",
    )
    clf.fit(X, y)

    preds = clf.predict(X[:10])
    assert preds.shape == (10,)
    assert set(preds).issubset(set(y))

    proba = clf.predict_proba(X[:10])
    assert proba.shape == (10, len(np.unique(y)))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-4)

    model_path = tmp_path / "nn_classifier.joblib"
    joblib.dump(clf, model_path)
    loaded = joblib.load(model_path)
    preds_loaded = loaded.predict(X[:5])
    assert preds_loaded.shape == (5,)
