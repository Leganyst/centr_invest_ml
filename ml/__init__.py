"""ML package exposing configuration, training utilities, and runtime classifier."""

from .model import TransactionClassifier
from .train import train_model

__all__ = ["TransactionClassifier", "train_model"]
