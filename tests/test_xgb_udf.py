import mlflow
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from sklearn import datasets

from snowflake.ml.mlflow.deploy.udf_util import upload_model_from_mlflow
from tests.conftest import ModelWithData, build_mlflow_model


@pytest.fixture(scope="module")
def xgb_model():
    """A XGBoost model trained directly using type `Booster`."""
    iris = datasets.load_iris()
    X = pd.DataFrame(
        iris.data[:, :2],
        columns=iris.feature_names[:2],  # we only take the first two features.
    )
    y = iris.target
    dtrain = xgb.DMatrix(X, y)
    model = xgb.train({"eval_metric": "auc"}, dtrain)
    return ModelWithData(model=model, x=X, predictions=model.predict(xgb.DMatrix(X)))


@pytest.fixture(scope="module")
def xgb_with_skl_api_model():
    """A XGBoost model trained using sklearn interface."""
    wine = datasets.load_wine()
    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = pd.Series(wine.target)
    model = xgb.XGBRegressor(n_estimators=10)
    model.fit(X, y)
    return ModelWithData(model=model, x=X, predictions=model.predict(X))


@pytest.fixture
def xgb_mlflow_model_path(xgb_model, tmp_path):
    """A MLflow model directory path given provided XGBoost native model."""
    return build_mlflow_model(xgb_model, tmp_path, mlflow.xgboost.log_model, package_requirements=["xgboost"])


@pytest.fixture
def xgb_with_skl_api_mlflow_model_path(xgb_with_skl_api_model, tmp_path):
    """A MLflow model directory path given provided XGBoost model using sklearn interface."""
    return build_mlflow_model(
        xgb_with_skl_api_model,
        tmp_path,
        mlflow.xgboost.log_model,
        package_requirements=["xgboost"],
    )


def _common_validations(
    session,
    model,
    model_path,
    udf_server,
    test_data_X=None,
    test_data_y=None,
):
    udf_name = "my_model_udf"
    upload_model_from_mlflow(
        session,
        model_dir_path=model_path,
        udf_name=udf_name,
        test_data_X=test_data_X,
        test_data_y=test_data_y,
    )

    udf_server.validate(
        session=session,
        model_path=model_path,
        model=model,
        udf_name=udf_name,
        packages=["xgboost"],
    )


def test_xgb_native_with_test_data(xgb_mlflow_model_path, xgb_model, udf_server):
    from snowflake.snowpark import Session

    session = Session()
    test_data_X = pd.DataFrame(np.array([[5.1, 3.5]]))
    test_data_y = pd.Series(np.array([0.01457067]))
    _common_validations(session, xgb_model, xgb_mlflow_model_path, udf_server, test_data_X, test_data_y)


def test_xgb_with_skl_api(xgb_with_skl_api_mlflow_model_path, xgb_with_skl_api_model, udf_server):
    from snowflake.snowpark import Session

    session = Session()
    _common_validations(session, xgb_with_skl_api_model, xgb_with_skl_api_mlflow_model_path, udf_server)


def test_xgb_native(xgb_mlflow_model_path, xgb_model, udf_server):
    from snowflake.snowpark import Session

    session = Session()
    _common_validations(session, xgb_model, xgb_mlflow_model_path, udf_server)
