from typing import List

# TODO: add any additional required imports here

from plexe.internal.models.entities.artifact import Artifact
from plexe.internal.models.interfaces.predictor import Predictor


class PredictorImplementation(Predictor):
    def __init__(self, artifacts: List[Artifact]):
        """
        Instantiates the predictor using the provided model artifacts.
        :param artifacts: list of BinaryIO artifacts
        """
        # TODO: add model loading code here; use _get_artifact helper to select the artifact by name
        # Example:
        # artifact = self._get_artifact("model", artifacts)
        # with artifact.get_as_handle() as binary_io:
        #     # Load the model from the handle
        #     # self.model = load_model(binary_io)

    def predict(self, inputs: dict) -> dict:
        """
        Given an input conforming to the input schema, return the model's prediction
        as a dict conforming to the output schema.
        """
        # TODO: add inference code here
        # Example: return self._postprocess_output(self.model.predict(self._preprocess_input(inputs)))
        pass

    def _preprocess_input(self, inputs: dict):
        """Map the input data from a dict to the input format of the underlying model."""
        # TODO: add input preprocessing code here
        pass

    def _postprocess_output(self, outputs) -> dict:
        """Map the output from the underlying model to a dict compliant with the output schema."""
        # TODO: add output postprocessing code here
        pass

    @staticmethod
    def _get_artifact(name: str, artifacts: List[Artifact]) -> Artifact:
        """Given the name of a binary artifact, return the corresponding artifact from the list."""
        # Do not modify this method.
        for artifact in artifacts:
            if artifact.name == name:
                return artifact
        raise ValueError(f"Artifact {name} not found in the provided artifacts.")


# REFERENCES:
# The Artifact class has the following relevant methods:
#
# class Artifact:
#     name: str
#
#     def get_as_handle(self) -> BinaryIO:
#         """
#         Get the artifact as a file-like object.
#         """
#         ...
#
# The Artifact always has a 'name' attribute, which should be used to identify the artifact. The internal definition
# of the Artifact class is not relevant here, except for the 'get_as_handle' method, which returns a file-like BinaryIO
# object. This should be used to access the artifact's data.
