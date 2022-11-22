import logging
import os
import tempfile
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.models import Model

from snowflake.ml.mlflow.deploy.constants import SKL_FLAVOR, XGB_FLAVOR
from snowflake.ml.mlflow.util.pkg_util import extract_package_requirements
from snowflake.snowpark import Session
from snowflake.snowpark.types import (
    BooleanType,
    ByteType,
    FloatType,
    IntegerType,
    PandasDataFrameType,
    PandasSeriesType,
    StringType,
)

_logger = logging.getLogger(__name__)
_MLMODEL_FILE_NAME = "MLmodel"
_REQUIREMENTS_TXT = "requirements.txt"
_SERIALIZATION_FORMAT_PICKLE = "pickle"
_SERIALIZATION_FORMAT_CLOUDPICKLE = "cloudpickle"
_SUPPORTED_SERIALIZATION_FORMATS = [
    _SERIALIZATION_FORMAT_PICKLE,
    _SERIALIZATION_FORMAT_CLOUDPICKLE,
]
_DTYPE_TYPE_MAPPING = {
    np.dtype("float64"): FloatType(),
    np.dtype("float32"): FloatType(),
    np.dtype("int64"): IntegerType(),
    np.dtype("bool"): BooleanType(),
    np.dtype("int32"): IntegerType(),
    np.dtype("str"): StringType(),
    np.dtype("bytes"): ByteType(),
}

# `extra_statements` is used for module level one time set up
# `load_expression` returns a model-like object with `predict(df: Dataframe)`
# `model_rel_path` is the relative path under model artifact directory
# `max_batch_size_string` is the max batch size
# `test_statements` used for model deploy regression test before start of inference
_UDF_CODE_TEMPLATE = """
import pandas as pd
import sys
from _snowflake import vectorized
{extra_statements}

IMPORT_DIRECTORY_NAME = "snowflake_import_directory"
import_dir = sys._xoptions[IMPORT_DIRECTORY_NAME]
with open(import_dir + '{model_rel_path}', "rb") as f:
    model = {load_expression}

@vectorized(input=pd.DataFrame, max_batch_size={max_batch_size_string})
def infer(df):
    ans = model.predict(df)
    return pd.Series(ans)

{test_statements}
"""

# `serialized_dx`: JSON serialized 2d pd.DataFrame
# `serialized_dy`: JSON serialized pd.Series
_MODEL_TEST_TEMPLATE = """
_test_X=pd.read_json('{serialized_dx}')
_test_y=pd.read_json('{serialized_dy}', typ='series')
assert isinstance(_test_X, pd.DataFrame)
assert isinstance(_test_y, pd.Series)

import numpy as np

np.testing.assert_almost_equal(_test_y.to_numpy(), infer(_test_X).to_numpy())
"""


class InferUDFHelperBase(ABC):
    def __init__(
        self, model_path: str, name: str, use_latest_package_version: bool, stage_location: Optional[str]
    ) -> None:
        self._model_path = model_path
        self._name = name
        model_config_path = os.path.join(self._model_path, _MLMODEL_FILE_NAME)
        self._model_conf = Model.load(model_config_path)
        requirements_txt_path = os.path.join(model_path, _REQUIREMENTS_TXT)
        self._pkg_requirements = _handle_package_versions(
            requirements_txt_path, use_latest_package_version=use_latest_package_version
        )
        self._stage_location = stage_location

    @abstractmethod
    def generate_udf(
        self,
        *,
        session: Session,
        max_batch_size: Optional[int],
        persist_udf_file: bool,
        test_data_X: Optional[pd.DataFrame],
        test_data_y: Optional[pd.Series],
    ):
        """Generate model backed UDF in Snowflake warehouse.

        Args:
            session (Session): A snowpark session.
            max_batch_size (Optional[int]): User provided max_batch_size configuration.
            persist_udf_file (bool): Whether to persist generated UDF file.
            test_data_X (Optional[pd.DataFrame]): Test input data as 2d dataframe.
            test_data_y (Optional[pd.Series]): Test prediction data as series.
        """
        pass

    @staticmethod
    def _autogen_model_infer_udf(
        session,
        model_signature,
        model_full_path: str,
        model_rel_path: str,
        udf_name: str,
        packages: List[str],
        stage_location: Optional[str],
        model_load_expression: str,
        extra_statements: str = "",
        max_batch_size=None,
        persist_udf_file=False,
        test_data_X=None,
        test_data_y=None,
    ):
        output_type = model_signature.outputs.numpy_types()[0]
        input_types = model_signature.inputs.numpy_types()
        snowpark_input_types = [_DTYPE_TYPE_MAPPING[dt] for dt in input_types]
        snowpark_output_type = _DTYPE_TYPE_MAPPING[output_type]
        max_batch_size_string = "None" if max_batch_size is None else f"{max_batch_size}"
        test_statements = ""
        if test_data_X is not None:
            serialized_dx = test_data_X.to_json()
            serialized_dy = test_data_y.to_json()
            test_statements = _MODEL_TEST_TEMPLATE.format(serialized_dx=serialized_dx, serialized_dy=serialized_dy)

        udf_code = _UDF_CODE_TEMPLATE.format(
            model_rel_path=model_rel_path,
            extra_statements=extra_statements,
            load_expression=model_load_expression,
            max_batch_size_string=max_batch_size_string,
            test_statements=test_statements,
        )
        is_permanent = False
        if stage_location:
            is_permanent = True
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=not persist_udf_file) as f:
            final_packages = ["pandas"]
            if packages:
                final_packages = final_packages + packages
            f.write(udf_code)
            f.flush()
            if persist_udf_file:
                _logger.info(f"Generated UDF file is persisted at: {f.name}")
            return session.udf.register_from_file(
                file_path=f.name,
                func_name="infer",
                name=f"{udf_name}",
                return_type=PandasSeriesType(snowpark_output_type),
                input_types=[PandasDataFrameType(snowpark_input_types)],
                replace=True,
                imports=[f"{model_full_path}"],
                packages=final_packages,
                stage_location=stage_location,
                is_permanent=is_permanent,
            )


class SKLUDFHelper(InferUDFHelperBase):
    def __init__(
        self, model_path: str, name: str, use_latest_package_version: bool, stage_location: Optional[str]
    ) -> None:
        super().__init__(
            model_path=model_path,
            name=name,
            use_latest_package_version=use_latest_package_version,
            stage_location=stage_location,
        )

    def _get_skl_model_info(self):
        flavor = self._model_conf.flavors[SKL_FLAVOR]
        model_artifact_path = os.path.join(self._model_path, flavor["pickled_model"])
        ser_format = flavor.get("serialization_format")
        if ser_format not in _SUPPORTED_SERIALIZATION_FORMATS:
            raise MlflowException("Only pickle and cloudpickle are supported as serialization format.")
        return (
            model_artifact_path,
            flavor["pickled_model"],
            ser_format,
        )

    def generate_udf(
        self,
        *,
        session: Session,
        max_batch_size: Optional[int],
        persist_udf_file: bool,
        test_data_X: Optional[pd.DataFrame],
        test_data_y: Optional[pd.Series],
    ):
        (
            model_full_path,
            model_relative_path,
            serialization_format,
        ) = self._get_skl_model_info()
        if serialization_format == _SERIALIZATION_FORMAT_CLOUDPICKLE:
            extra_statements = "import cloudpickle"
            load_expression = "cloudpickle.load(f)"
        else:
            extra_statements = ""
            load_expression = "pickle.load(f)"

        self._autogen_model_infer_udf(
            session=session,
            model_signature=self._model_conf.signature,
            model_full_path=model_full_path,
            model_rel_path=model_relative_path,
            udf_name=self._name,
            packages=self._pkg_requirements,
            stage_location=self._stage_location,
            model_load_expression=load_expression,
            extra_statements=extra_statements,
            max_batch_size=max_batch_size,
            persist_udf_file=persist_udf_file,
            test_data_X=test_data_X,
            test_data_y=test_data_y,
        )


class XGBUDFHelper(InferUDFHelperBase):
    def __init__(
        self, model_path: str, name: str, use_latest_package_version: bool, stage_location: Optional[str]
    ) -> None:
        super().__init__(
            model_path=model_path,
            name=name,
            use_latest_package_version=use_latest_package_version,
            stage_location=stage_location,
        )

    def _get_model_info(self):
        flavor = self._model_conf.flavors[XGB_FLAVOR]
        model_class = flavor.get("model_class", "xgboost.core.Booster")
        model_artifact_path = os.path.join(self._model_path, flavor["data"])
        return (model_artifact_path, flavor["data"], model_class)

    def generate_udf(
        self,
        *,
        session: Session,
        max_batch_size: Optional[int],
        persist_udf_file: bool,
        test_data_X: Optional[pd.DataFrame],
        test_data_y: Optional[pd.Series],
    ):
        (
            model_full_path,
            model_relative_path,
            model_class,
        ) = self._get_model_info()

        extra_statements = f"""
import importlib
import xgboost as xgb
def _get_class_from_string(fully_qualified_class_name):
    module, class_name = fully_qualified_class_name.rsplit(".", maxsplit=1)
    return getattr(importlib.import_module(module), class_name)

model_instance = _get_class_from_string("{model_class}")()

def wrapper(model_path):
    model_instance.load_model(f.name)
    class _Wrapper:
        def __init__(self, model):
            self._model = model

        def predict(self, df):
            if isinstance(self._model, xgb.Booster):
                return self._model.predict(xgb.DMatrix(df), validate_features=False)
            else:
                return self._model.predict(df)

    return _Wrapper(model=model_instance)
        """

        load_expression = """wrapper(f.name)
        """
        self._autogen_model_infer_udf(
            session=session,
            model_signature=self._model_conf.signature,
            model_full_path=model_full_path,
            model_rel_path=model_relative_path,
            udf_name=self._name,
            packages=self._pkg_requirements,
            stage_location=self._stage_location,
            model_load_expression=load_expression,
            extra_statements=extra_statements,
            max_batch_size=max_batch_size,
            persist_udf_file=persist_udf_file,
            test_data_X=test_data_X,
            test_data_y=test_data_y,
        )


def _udf_helper_factory(
    flavors: Dict[str, Any],
    mlflow_model_dir_path: str,
    udf_name: str,
    use_latest_package_version: bool,
    stage_location: Optional[str],
) -> InferUDFHelperBase:
    if not flavors:
        raise MlflowException("No flavor is specified for model deployment.")
    if SKL_FLAVOR in flavors:
        return SKLUDFHelper(mlflow_model_dir_path, udf_name, use_latest_package_version, stage_location)
    elif XGB_FLAVOR in flavors:
        return XGBUDFHelper(mlflow_model_dir_path, udf_name, use_latest_package_version, stage_location)
    else:
        raise MlflowException(f"{flavors.keys()} flavors are not supported.")


def _handle_package_versions(requirements_path: str, use_latest_package_version: bool) -> List[str]:
    requirements_lines = open(requirements_path).read().splitlines()
    try:
        return extract_package_requirements(
            requirements_lines=requirements_lines,
            exclusions={"mlflow", "mypy-extensions"},
            use_latest=use_latest_package_version,
        )
    except ValueError as e:
        raise MlflowException(str(e))


def upload_model_from_mlflow(
    session: Session,
    *,
    model_dir_path: str,
    udf_name: str,
    max_batch_size: Optional[int] = None,
    persist_udf_file: bool = True,
    test_data_X: Optional[pd.DataFrame] = None,
    test_data_y: Optional[pd.Series] = None,
    use_latest_package_version: bool = True,
    stage_location: Optional[str] = None,
):
    """Upload a supported model packaged by MLflow to be deployed as Snowflake Python UDF.

    The model file is automatically uploaded to stage from local temporary storage as part of the UDF registration.

    Args:
        session (Session): A Snowflake session object.
        model_dir_path (str): The model directory generated by MLflow run.
        udf_name (str): Name for the UDF.
        max_batch_size (int, optional): Maximum batch sizes for a single vectorized UDF invocation.
            The size is not guaranteed by Snowpark. Default to be None.
        persist_udf_file (bool, optional): Whether to keep the UDF file generated.
            Default to be False.
        test_data_X (pd.DataFrame, optional): 2d dataframe used as input test data.
        test_data_y (pd.Series, optional): 1d series used as expected prediction results.
            During testing, model predictions are compared with the expected predictions given in `test_data_y`.
            For comparing regression results, 1e-7 precision is used.
        use_latest_package_version (bool, optional):
            Whether to use latest package versions available in Snowlfake conda channel.
            Defaults to True. Settomg this flag to True will greatly increase the chance of successful deployment
            of the model to the Snowflake warehouse.
        stage_location (str, optional): Stage location to store the UDF and dependencies(format: `@my_named_stage`).
            It can be any stage other than temporary stages and external stages. If not specified,
            UDF deployment is temporary and tied to the session. Default to be none.
    """
    model_config_path = os.path.join(model_dir_path, _MLMODEL_FILE_NAME)
    if not os.path.exists(model_config_path):
        raise MlflowException("Model config file did not exist.")
    model_conf = Model.load(model_config_path)
    if model_conf.signature is None:
        raise MlflowException("Signature is required to automatically generate the UDF.")
    if len(model_conf.signature.outputs.numpy_types()) != 1:
        raise MlflowException("Only single model output is supported.")
    is_test_data_X_set = test_data_X is not None
    is_test_data_y_set = test_data_y is not None
    if is_test_data_X_set and is_test_data_y_set:
        if test_data_X.shape[0] != test_data_y.shape[0]:
            raise MlflowException("Test data dimension mis-match.")
    elif is_test_data_X_set != is_test_data_y_set:
        raise MlflowException("Both input & output test data needs to be specified.")
    _udf_helper_factory(
        model_conf.flavors,
        model_dir_path,
        udf_name,
        use_latest_package_version,
        stage_location,
    ).generate_udf(
        session=session,
        max_batch_size=max_batch_size,
        persist_udf_file=persist_udf_file,
        test_data_X=test_data_X,
        test_data_y=test_data_y,
    )
