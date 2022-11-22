from mlflow.exceptions import MlflowException

from snowflake.ml.mlflow.deploy.deployment_client import SnowflakeDeploymentClient


def run_local(name, model_uri, flavor=None, config=None):
    """
    Deploys the specified model locally for testing. Note that models deployed locally cannot
    be managed by other deployment APIs (e.g. ``update_deployment``, ``delete_deployment``, etc).
    """
    raise MlflowException("`run_local` is not supported.")


def target_help():
    """
    Return a string containing detailed documentation on the current deployment target
    to be displayed when users invoke the ``mlflow deployments help -t <target-name>`` CLI command.
    This method should be defined within the module specified by the plugin author.
    """
    return """
    For the Snowflake deployment plugin the `target_uri` needs to have the`snowflake` scheme.
    Connection parameters have to be specified by adding `?connection={CONNECTION_NAME}`.
    The `CONNECTION_NAME` references the connection specified in the SnowSQL configuration file
    e.g. `snowflake:/?connection=connections.ml`

    For `create_deployment`, there's a list of available configuration options:
        max_batch_size (int): Max batch size for a single vectorized UDF invocation.
            The size is not guaranteed by Snowpark.
        persist_udf_file (bool): Whether to keep the UDF file generated.
        test_data_X (pd.DataFrame): 2d dataframe used as input test data.
        test_data_y (pd.Series): 1d series used as expected prediction results.
            During testing, model predictions are compared with the expected predictions given in `test_data_y`.
            For comparing regression results, 1e-7 precision is used.
        use_latest_package_version (bool): Whether to use latest package versions available in Snowlfake conda channel.
            Defaults to True. Settomg this flag to True will greatly increase the chance of successful deployment
            of the model to the Snowflake warehouse.
        stage_location (str, optional): Stage location to store the UDF and dependencies(format: `@my_named_stage`).
            It can be any stage other than temporary stages and external stages. If not specified,
            UDF deployment is temporary and tied to the session. Default to be none.

    `update_deployment` is not supported. To update an existing deployment use `create_deployment` with the same
    deployment name.
    """


__all__ = ["SnowflakeDeploymentClient", "run_local", "target_help"]
