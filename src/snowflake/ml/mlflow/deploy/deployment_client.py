from enum import Enum, unique
from typing import Any, Dict, Set

import pandas as pd
from mlflow.deployments import BaseDeploymentClient
from mlflow.exceptions import MlflowException

# TODO(halu): Revisit `_download_artifact_from_uri`
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.annotations import experimental

from snowflake.ml.mlflow.deploy.constants import SUPPORTED_FLAVORS
from snowflake.ml.mlflow.deploy.management_util import DeploymentHelper
from snowflake.ml.mlflow.deploy.udf_util import upload_model_from_mlflow
from snowflake.ml.mlflow.session_util import _resolve_session
from snowflake.ml.mlflow.util.telemetry import method_session_usage_logging_helper
from snowflake.snowpark import functions as F


@unique
class ConfigField(Enum):
    MAX_BATCH_SIZE = "max_batch_size"
    PERSIST_UDF_FILE = "persist_udf_file"
    TEST_DATA_X = "test_data_X"
    TEST_DATA_Y = "test_data_y"
    USE_LATEST_PACKAGE_VERSION = "use_latest_package_version"
    STAGE_LOCAION = "stage_location"


class SnowflakeDeploymentClient(BaseDeploymentClient):
    """
    Initialize a deployment client for Snowflake. The Snowpark session can be explicitly
    provided via `set_session/create_session`. If no session is given, then a
    session will be created using the `connection` query parameter from the
    `target_uri`.

    target_uri: A URI that follows one of the following formats:
        * `snowflake`: This is sufficient if a session is provided explicitly.
        * `snowflake://?connection={connection_name}`: This will try to load the local SnowSQL
            configuration files and use `connection_name` as default session. Reference could be
            found here: https://docs.snowflake.com/en/user-guide/snowsql-config.html
    """

    def __init__(self, target_uri) -> None:
        super().__init__(target_uri)
        self._session = _resolve_session(target_uri)
        self._validate_session()
        self._deploy_helper = DeploymentHelper(self._session)

    def parse_config(self, config) -> Dict[str, Any]:
        """Parse input deployment configuration to conform to valid config options."""
        if not config:
            return dict()
        parsed = dict()
        for field in ConfigField:
            if field.value in config:
                parsed[field.value] = config[field.value]
        return parsed

    def usage_logging(fields_to_log: Set[str]):
        """Decorator to send client usage telemetry data."""

        def extract_session(captured):
            assert isinstance(
                captured, SnowflakeDeploymentClient
            ), f"Should capture instance of `SnowflakeDeploymentClient`, but got type {type(captured)}."
            return captured._session

        return method_session_usage_logging_helper(fields_to_log=fields_to_log, sesssion_extractor=extract_session)

    @experimental
    @usage_logging(fields_to_log={"name", "config"})
    def create_deployment(self, name, model_uri, flavor=None, config=None, endpoint=None):
        """Deploy the model to Snowflake as a UDF.
        This method blocks until the deployment operation is completed.

        Args:
            name: Unique name to use for the deployment.
            model_uri : URI of the model to deploy.
            flavor (optional): Model flavor to deploy. If unspecified, will infer
                based on model metadata. Defaults to None.
            config (optional): Snowflake-specific configuration for the deployment.
                Defaults to None.
            endpoint (optional): Ignored for now.

        Raises:
            MlflowException: Raise if specified flavor is not supported.
        """
        if flavor and flavor not in SUPPORTED_FLAVORS:
            raise MlflowException(f"Only {SUPPORTED_FLAVORS} flavors are supported now.")
        model_path = _download_artifact_from_uri(model_uri)
        parsed_config = self.parse_config(config)
        upload_model_from_mlflow(self._session, model_dir_path=model_path, udf_name=name, **parsed_config)
        self._deploy_helper.tag_deployment(name=name)

    @experimental
    @usage_logging(fields_to_log={"name", "config"})
    def delete_deployment(self, name, config=None, endpoint=None):
        """Delete the deployment with name `name` from Snowflake.
        Deletion is idempotent.

        Args:
            name: Name of deployment to delete.
            config (optional): Snowflake specific configuration for deployment.
                Ignored for now.
            endpoint (optional): Ignored for now. Defaults to None.
        """
        self._deploy_helper.delete_deployment(name=name)

    @experimental
    @usage_logging(fields_to_log={"name"})
    def get_deployment(self, name, endpoint=None):
        """Returns a dictionary describing the specified deployment created by the plugin
        under specified database/schema/role.

        Args:
            name: Name of deployment to get.
            endpoint (optional): Ignored for now. Defaults to None.

        Returns:
            A dict containing
                1) `name` of the deployment
                2) `signature` of the generated Snowflake UDF function.
        """
        return {
            "name": name,
            "signature": self._deploy_helper.get_deployment(name=name),
        }

    @experimental
    @usage_logging(fields_to_log={})
    def list_deployments(self, endpoint=None):
        """Return an unpaginated list of deployments created by the plugin under specified database/schema/role.

        Args:
            endpoint (optional): Ignored for now. Defaults to None.

        Returns:
            List of dicts, each containing:
                1) `name` of the deployment
                2) `signature` of the generated Snowflake UDF function.
        """
        return [{"name": name, "signature": signature} for (name, signature) in self._deploy_helper.list_deployments()]

    @experimental
    @usage_logging(fields_to_log={"deployment_name"})
    def predict(self, deployment_name=None, df=None, endpoint=None):
        """Compute predictions on the given dataframe using the specified deployment created by the plugin
        under specified database/schema/role.

        Args:
            deployment_name: Name of the deployment.
            df: Either pandas DataFrame or Snowpark DataFrame.
            endpoint (optional): Ignored for now. Defaults to None.

        Returns:
            Result as a pandas DataFrame.

        Raises:
            MlflowException: Raise if `deployment_name` is not specified.
            MlflowException: Raise if `df` is not specified.
        """
        if not deployment_name:
            raise MlflowException("deployment_name is missing.")
        if not df:
            raise MlflowException("df is missing.")
        if isinstance(df, pd.DataFrame):
            df = self.session.create_dataframe(df)
        return df.select(F.call_udf(deployment_name, *[F.col(x) for x in df.columns])).to_pandas()

    @experimental
    def update_deployment(self, name, model_uri=None, flavor=None, config=None, endpoint=None):
        """Updating a deployment is not supported."""
        raise MlflowException("update_deployment is not supported. Please use create_deployment directly.")

    def _validate_session(self):
        assert self._session is not None, "A valid Snowpark session is required. "
        "Use `set_session` or `create_session` to associate a Snowpark session with this deployment client. "
        "Alternatively, the local SnowSQL configuration can be referenced in the `target_uri` to specify a connection."
        if not self._session.get_current_database():
            raise MlflowException("Missing DATABASE from session.")
        if not self._session.get_current_schema():
            raise MlflowException("Missing SCHEMA from session.")
