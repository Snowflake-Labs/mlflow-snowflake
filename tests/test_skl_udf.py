import mlflow
import numpy as np
import pandas as pd
import pytest

from snowflake.ml.mlflow.deploy.udf_util import upload_model_from_mlflow
from tests.conftest import ModelWithData, build_mlflow_model


@pytest.fixture(scope="module")
def skl_model():
    """A naive sklearn trained model with module scope."""
    import sklearn.linear_model as glm
    from sklearn import datasets

    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features.
    y = iris.target
    linear_lr = glm.LogisticRegression()
    linear_lr.fit(X, y)
    return ModelWithData(model=linear_lr, x=X, predictions=linear_lr.predict(X))


@pytest.fixture
def skl_mlflow_model_path(skl_model, tmp_path):
    """A MLflow model directory path given provided sklearn model."""
    return build_mlflow_model(
        skl_model,
        tmp_path,
        mlflow.sklearn.log_model,
        package_requirements=["scikit-learn"],
    )


def test_skl(skl_mlflow_model_path, skl_model, udf_server):
    """E2E test to make sure model upload works for sklearn model."""
    from snowflake.snowpark import Session

    session = Session()
    udf_name = "my_model_udf"
    upload_model_from_mlflow(session, model_dir_path=skl_mlflow_model_path, udf_name=udf_name)

    udf_server.validate(
        session=session,
        model_path=skl_mlflow_model_path,
        model=skl_model,
        udf_name=udf_name,
        packages=["scikit-learn"],
    )


def test_skl_with_correct_test_data(skl_mlflow_model_path, skl_model, udf_server):
    """E2E test with test data assertion passing."""
    from snowflake.snowpark import Session

    session = Session()
    udf_name = "my_model_udf"
    test_data_X = pd.DataFrame(np.array([[5.9, 3.0]]))
    test_data_y = pd.Series(np.array([1]))
    upload_model_from_mlflow(
        session=session,
        model_dir_path=skl_mlflow_model_path,
        udf_name=udf_name,
        test_data_X=test_data_X,
        test_data_y=test_data_y,
    )

    udf_server.validate(
        session=session,
        model_path=skl_mlflow_model_path,
        model=skl_model,
        udf_name=udf_name,
        packages=["scikit-learn"],
    )


def test_skl_with_wrong_test_data(skl_mlflow_model_path, skl_model, udf_server):
    """Expect raise if test data could not match during inference setup."""
    from snowflake.snowpark import Session

    session = Session()
    udf_name = "my_model_udf"
    test_data_X = pd.DataFrame(np.array([[5.9, 3.0]]))
    test_data_y = pd.Series(np.array([0]))
    upload_model_from_mlflow(
        session=session,
        model_dir_path=skl_mlflow_model_path,
        udf_name=udf_name,
        test_data_X=test_data_X,
        test_data_y=test_data_y,
    )

    with pytest.raises(AssertionError, match=".*Arrays are not almost equal.*"):
        udf_server.validate(
            session=session,
            model_path=skl_mlflow_model_path,
            model=skl_model,
            udf_name=udf_name,
            packages=["scikit-learn"],
        )
