"""Model loading and prediction, independent of the web framework."""

from sentimetro.inference.predictor import Prediction, Predictor, load_predictor

__all__ = ["Prediction", "Predictor", "load_predictor"]
