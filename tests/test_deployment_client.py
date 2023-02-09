from unittest.mock import MagicMock

import pytest
from mlflow.exceptions import MlflowException

from snowflake.ml.mlflow import session_util as util
from snowflake.ml.mlflow.deploy.deployment_client import (
    ConfigField,
    SnowflakeDeploymentClient,
)


@pytest.fixture
def mock_deployer(monkeypatch):
    deployer = MagicMock()
    monkeypatch.setattr("snowflake.ml.mlflow.deploy.deployment_client.DeploymentHelper", deployer)
    return deployer


@pytest.fixture
def mock_send_usage(monkeypatch):
    client = MagicMock()
    monkeypatch.setattr("snowflake.ml.mlflow.util.telemetry.TelemetryClient", client)
    su = MagicMock()
    client().send_usage = su
    return su


def test_usage_logging(clean_session, mock_deployer, mock_send_usage):
    util.set_session(clean_session)
    client = SnowflakeDeploymentClient("snowflake")
    client.get_deployment("udf1")
    mock_send_usage.assert_called_once
    assert mock_send_usage.call_args.args[0] == "get_deployment"
    assert mock_send_usage.call_args.args[1] == {"name": "'udf1'"}
    assert mock_send_usage.call_args.args[2] is None


def test_parse_config(clean_session):
    """Expect extra configurations filtered."""
    c1 = {
        "missing1": 1,
        ConfigField.MAX_BATCH_SIZE.value: 2,
        ConfigField.USE_LATEST_PACKAGE_VERSION.value: False,
    }
    util.set_session(clean_session)
    client = SnowflakeDeploymentClient("snowflake")
    res = client.parse_config(c1)
    assert res == {
        ConfigField.MAX_BATCH_SIZE.value: 2,
        ConfigField.USE_LATEST_PACKAGE_VERSION.value: False,
    }


def test_parse_config_when_none(clean_session):
    """Expect empty dict."""
    util.set_session(clean_session)
    client = SnowflakeDeploymentClient("snowflake")
    res = client.parse_config(None)
    assert res == {}


def test_create_deployment_success(clean_session, monkeypatch):
    """Expect return dict containing `name` attribute."""
    util.set_session(clean_session)
    monkeypatch.setattr("snowflake.ml.mlflow.deploy.deployment_client.upload_model_from_mlflow", MagicMock())
    monkeypatch.setattr("snowflake.ml.mlflow.deploy.deployment_client._download_artifact_from_uri", MagicMock())
    client = SnowflakeDeploymentClient("snowflake")
    res_dict = client.create_deployment("modelx", "uri", flavor="sklearn")
    assert "name" in res_dict and res_dict["name"] == "modelx"
    assert "udf_name" in res_dict and res_dict["udf_name"] == "MLFLOW$MODELX"


def test_predict_with_missing_arguments(clean_session):
    """Expect raise when missing required arguments."""
    util.set_session(clean_session)
    client = SnowflakeDeploymentClient("snowflake")
    with pytest.raises(MlflowException, match="deployment_name is missing."):
        client.predict(deployment_name=None, df=None)
    with pytest.raises(MlflowException, match="df is missing."):
        client.predict(deployment_name="udf", df=None)


def test_update_deployment_not_supported(clean_session):
    util.set_session(clean_session)
    client = SnowflakeDeploymentClient("snowflake")
    with pytest.raises(MlflowException, match=r"update_deployment is not supported.+"):
        client.update_deployment("name")


def test_validate_session_with_missing_database(clean_session):
    util.set_session(clean_session)
    clean_session.get_current_database.return_value = None
    with pytest.raises(MlflowException, match="Missing DATABASE from session."):
        SnowflakeDeploymentClient("snowflake")


def test_validate_session_with_missing_schema(clean_session):
    util.set_session(clean_session)
    clean_session.get_current_schema.return_value = None
    with pytest.raises(MlflowException, match="Missing SCHEMA from session."):
        SnowflakeDeploymentClient("snowflake")
