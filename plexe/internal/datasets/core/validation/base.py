from abc import ABC, abstractmethod


class BaseDataValidator(ABC):
    @abstractmethod
    def validate(
        self,
        report_output_path: str,
        synthetic_data_path: str,
        reference_data_path: str = None,
        data_schema: dict = None,
    ) -> str:
        """
        Validate synthetic data against reference data and schema. The validation results are saved to a report
        jupyter notebook.
        :param report_output_path: path to save the validation report
        :param synthetic_data_path: path to the synthetic data to validate
        :param reference_data_path: if provided, path to the reference data to compare against
        :param data_schema: if provided, schema of the data to validate against
        :return: path to the validation report
        """
        pass
