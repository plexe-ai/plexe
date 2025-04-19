import plexe


class PredictionService:
    """Service for making predictions with trained models"""

    _model_cache = {}

    @classmethod
    def predict(cls, model_id, data):
        """Make a prediction using a trained model"""
        # Try to get model from cache
        if model_id not in cls._model_cache:
            # Load model from disk
            model = plexe.load_model(f"model-{model_id}")
            cls._model_cache[model_id] = model
        else:
            model = cls._model_cache[model_id]

        # Make prediction
        return model.predict(data)
