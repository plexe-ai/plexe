import pandas as pd

# TODO: add any additional required imports here

from plexe.core.interfaces.feature_transformer import FeatureTransformer


class FeatureTransformerImplementation(FeatureTransformer):

    def transform(self, inputs: pd.DataFrame) -> pd.DataFrame:
        """
        Given a DataFrame representing a raw dataset, applies feature transformations to the
        dataset and returns the transformed DataFrame suitable for training an ML model.
        """
        # TODO: add feature transformation code here
        # Example: group by 'category' and sum 'value'
        # transformed_df = inputs.groupby('category')['value'].sum().reset_index()
        # return transformed_df
