import importlib.machinery
import importlib.util
import os
import sys
from abc import ABC, abstractmethod
from typing import Any, Callable, List, NamedTuple
from unittest.mock import MagicMock

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.models import Model, infer_signature
from mlflow.models.model import ModelInfo
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.file_utils import TempDir

from snowflake.ml.mlflow import session_util as util

MODLE_NAME = "tst_model"


def _load_module_from_file(name: str, path: str):
    """Load module from a python file."""
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    loaded = importlib.util.module_from_spec(spec)
    loader.exec_module(loaded)
    return loaded


def _new_module_from_source_string(name: str, source: str) -> None:
    spec = importlib.util.spec_from_loader(name, loader=None)
    module = importlib.util.module_from_spec(spec)
    exec(source, module.__dict__)
    sys.modules[name] = module
    globals()[name] = module


_VECTORIZED_DECORATOR_TEMPLATE = """
import functools

def vectorized(func=None, *args0, **kwargs0):
    if func is None:
        return functools.partial(vectorized, *args0, **kwargs0)
    @functools.wraps(func)
    def wrapper(*args1, **kwargs1):
        return func(*args1, **kwargs1)
    return wrapper
"""


@pytest.fixture()
def session(monkeypatch):
    """A Mock snowpark session object."""
    sess = MagicMock()
    monkeypatch.setattr("snowflake.snowpark.Session", sess)
    return sess


@pytest.fixture
def clean_session(session, monkeypatch):
    """Always clear the session for each test set up."""
    builder = MagicMock()
    monkeypatch.setattr("snowflake.snowpark.session.Session.builder", builder)
    util._clear_session()
    return session


class ModelWithData(NamedTuple):
    # Actual ML model trained by some framework
    model: Any
    # Some input data
    x: Any
    # Predictions based on `x` used for UDF validations
    predictions: Any


class IUDFServer(ABC):
    @abstractmethod
    def validate(
        self,
        *,
        session,
        model_path: str,
        model: ModelWithData,
        udf_name: str,
        packages: List[str],
    ) -> None:
        """Validate UDF based on the expected information.

        Args:
            session (Session): A mock snowpark session
            model_path (str): MLflow model directory
            model (ModelWithData): Actual ML model
            udf_name (str): Name for the UDF
            packages (List[str]): Expected packages to include
        """
        pass


# TODO(halu): Add `max_batch_size` support
# TODO(halu): Add data type matching support
@pytest.fixture()
def udf_server(session) -> IUDFServer:
    """A fake UDF server fixture for UDF testing.

    The fixture will set up a simple environment in the process
    1) To run the UDF to verify predictions correctness by using pre-defined
    `x` and `predictions` from `model`(ModelWithData)
    2) Verify basic UDF metdata is correct
    Use it when your test needs to execute UDF and verify input & output matches.
    The fixture exposes interface `validate` for the execution and validation.
    See `IUDFServer` for detailed interface specification.

    Example:
        def test_x(udf_server):
            ...
            # this will run the UDF and assert the expectations.
            udf_server.validate(...)


    Returns:
        IUDFServer: A fixture object could provide `validate`
            to validate UDF execution and expected params
    """

    class _MockUDFServer(IUDFServer):
        def validate(
            self,
            *,
            session,
            model_path: str,
            tmp_path: str,
            model: ModelWithData,
            udf_name: str,
            packages: List[str],
        ):
            session.udf.register_from_file.assert_called_once()
            cargs = session.udf.register_from_file.call_args
            assert set(cargs.kwargs["packages"]) == set(["pandas", "mlflow", "filelock"] + packages)
            assert cargs.kwargs["name"] == udf_name
            assert len(cargs.kwargs["imports"]) == 1
            # set up the file
            udf_code_path = cargs.kwargs["file_path"]
            IMPORT_DIRECTORY_NAME = "snowflake_import_directory"
            # set up the zipfile
            import zipfile

            with zipfile.ZipFile(os.path.join(tmp_path, f"{MODLE_NAME}.zip"), "w", zipfile.ZIP_DEFLATED) as zipf:
                for root, _dirs, files in os.walk(model_path):
                    for file in files:
                        zipf.write(
                            os.path.join(root, file),
                            os.path.relpath(os.path.join(root, file), os.path.join(model_path, "..")),
                        )
            # set up path
            sys._xoptions[IMPORT_DIRECTORY_NAME] = tmp_path
            # set up the module
            _new_module_from_source_string("_snowflake", _VECTORIZED_DECORATOR_TEMPLATE)
            infer_module = _load_module_from_file("infer_module", udf_code_path)
            np.testing.assert_equal(model.predictions, infer_module.infer(pd.DataFrame(model.x)))

    return _MockUDFServer()


def build_mlflow_model(
    model: ModelWithData,
    tmp_path: str,
    log_func: Callable[..., ModelInfo],
    package_requirements: List[str],
) -> str:
    """Builds a provided model under MLflow and returns the directory path.

    Args:
        model (ModelWithData): A model trained by some framework
        tmp_path (str): A temp dir path
        log_func (Callable[..., ModelInfo]): Specific MLflow flavor function to log the model

    Returns:
        str: MLflow packaged model directory
    """
    with TempDir(chdr=True, remove_on_exit=True):
        with mlflow.start_run():
            ARTIFACT_PATH = MODLE_NAME
            sig = infer_signature(model.x, model.predictions)
            model_info = log_func(
                model.model,
                artifact_path=ARTIFACT_PATH,
                signature=sig,
                pip_requirements=package_requirements,
            )
            model_uri = "runs:/{run_id}/{artifact_path}".format(
                run_id=mlflow.active_run().info.run_id, artifact_path=ARTIFACT_PATH
            )
            assert model_info.model_uri == model_uri
            _download_artifact_from_uri(artifact_uri=model_uri, output_path=tmp_path)
            model_path = f"{tmp_path}/{ARTIFACT_PATH}/"
            # sanity check correctness
            Model.load(os.path.join(model_path, "MLmodel"))
            return model_path


@pytest.fixture(autouse=True)
def mock_snow_channel(monkeypatch):
    """Auto use to avoid actual subprocess.run against conda."""
    res = MagicMock()
    monkeypatch.setattr(
        "snowflake.ml.mlflow.util.pkg_util.check_compatibility_with_snowflake_conda_channel",
        res,
    )
    return res
