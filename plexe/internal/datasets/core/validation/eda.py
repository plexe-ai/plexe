import os
import subprocess

import nbformat as nbf

from .base import BaseDataValidator


class EdaDataValidator(BaseDataValidator):
    def validate(
        self,
        report_output_path: str,
        synthetic_data_path: str,
        reference_data_path: str = None,
        data_schema: dict = None,
    ) -> str:
        """
        Validates the synthetic data by comparing it to reference data (if available) and producing an
        EDA report as a Jupyter notebook. Includes distribution plots, null checks, and statistical comparisons.

        :param report_output_path: path to save the generated report
        :param synthetic_data_path: path to the synthetic data
        :param reference_data_path: path to the reference data
        :param data_schema:
        :return: path to the generated report
        """

        # load existing notebook from template path
        with open("resources/templates/eda-notebook-tabular.ipynb") as f:
            nb = nbf.read(f, as_version=4)

        # todo currently schema is not used, but it should be part of the validation report

        # string interpolation to fill the data paths into the template notebook
        for cell in nb["cells"]:
            cell["source"] = (
                str(cell["source"])
                .replace("{{$synthetic_data_path}}", os.path.basename(synthetic_data_path))
                .replace("{{$real_data_path}}", f"./../{reference_data_path}")
            )

        # save the notebook to the output path
        with open(report_output_path, "w") as f:
            nbf.write(nb, f)

        # attempt to execute the notebook and convert it to pdf
        try:
            # execute the notebook
            subprocess.run(
                ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", report_output_path], check=True
            )
            # convert the notebook to pdf
            subprocess.run(["jupyter", "nbconvert", "--to", "pdf", report_output_path], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing the notebook or converting it to pdf: {e}")

        return report_output_path
