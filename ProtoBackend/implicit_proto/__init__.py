from .dataset import SentenceDataset
from .encoder import PrototypeEncoder
from .prototype_builder import PrototypeBuilder
from .inference import ImplicitAspectDetector
from .train import train_prototypes
from .test import predict_aspects

__all__ = [
    "SentenceDataset",
    "PrototypeEncoder",
    "PrototypeBuilder",
    "ImplicitAspectDetector",
    "train_prototypes",
    "predict_aspects",
]
