# Snowflake MLflow Plugins

Currently provides an experimental Snowflake Mlflow Deployment Plugin.
This enables Mlflow users to deploy external trained Mlflow packaged models to Snowflake easily.

This plugin implements the [Python API](https://www.mlflow.org/docs/latest/python_api/mlflow.deployments.html)
and [CLI](https://www.mlflow.org/docs/latest/cli.html#mlflow-deployments) for MLflow deployment plugins.

## Usage
### Prerequisite
* The plugin relies on local anaconda installation to check if model dependencies could be satisfied. Suggest install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for minimum dependencies.

### Installation 
Please find latest release version [here](https://github.com/Snowflake-Labs/mlflow-snowflake/releases) to download latest `wheel`.
`pip install <local_path_to_wheel>` could install the package with name `snowflake-mlflow` for you.

### Session connection
Two ways of connection are supported to establish a Snowflake session for model deployment.
#### Python API
```python
from snowflake.mlflow import create_session
from mlflow.deployments import get_deploy_client
connection_parameters = dict()
create_session(connection_parameters)
target_uri = 'snowflake'
deployment_client = get_deploy_client(target_uri)
```
#### SnowSQL Configuration file
[SnowSQL Configuration file](https://docs.snowflake.com/en/user-guide/snowsql-config.html) is a familiar concept among existing SnowSQL CLI users and a neccessary way to establish connection to Snowflake if you intend to use MLflow CLI for model deployment.
For the Snowflake deployment plugin, the `target_uri` needs to have the`snowflake` scheme.
Connection parameters can be specified by adding `?connection={CONNECTION_NAME}`.
The `CONNECTION_NAME` references the connection specified in the SnowSQL configuration file e.g. `snowflake:/?connection=connections.ml`.
### Supported APIs
Following APIs are supported by both Python and CLI.
| Python API | CLI |
| ---| --- |
| `create_deployment(name, model_uri, flavor, config)`  | `mlflow deployments create`  |
| `delete_deployment(name)`  | `mlflow deployments delete`  |
| `get_deployment(name)` | `mlflow deployments get`  |
| `list_deployments` | `mlflow deployments list`  |
| `predict(deployment_name, df)` | `mlflow deployments predict`  |

*  `create_deployment`
```markdown
 Args:
     name: Unique name to use for the deployment.
     model_uri : URI of the model to deploy.
     flavor (optional): Model flavor to deploy. If unspecified, will infer
         based on model metadata. Defaults to None.
     config (optional): Snowflake-specific configuration for the deployment.
         Defaults to None.
         Detailed configuration options:
            max_batch_size (int): Max batch size for a single vectorized UDF invocation.
                The size is not guaranteed by Snowpark.
            persist_udf_file (bool): Whether to keep the UDF file generated.
            test_data_X (pd.DataFrame): 2d dataframe used as input test data.
            test_data_y (pd.Series): 1d series used as expected prediction results.
                During testing, model predictions are compared with the expected predictions given in `test_data_y`.
                For comparing regression results, 1e-7 precision is used.
            use_latest_package_version (bool): Whether to use latest package versions available in Snowlfake conda channel.
                Defaults to True. Set this flag to True will greatly increase the chance of successful deployment
                of the model to the Snowflake warehouse.
            stage_location (str, optional): Stage location to store the UDF and dependencies(format: `@my_named_stage`).
                It can be any stage other than temporary stages and external stages. If not specified,
                UDF deployment is temporary and tied to the session. Default to be none.
```
Detailed configuration options for `create_deployment` could also be retrieved by  `mlflow deployments help -t snowflake`

* `predict`
```
Args:
    deployment_name: Name of the deployment.
    df: Either pandas DataFrame or Snowpark DataFrame.

Returns:
    Result as a pandas DataFrame.
```

### Deployment Name Convention In SQL
To use deployed model in Snowflake SQL context with name `my_model_x`, you could invoke with `MLFLOW$` prefix:
```sql
SELECT MLFLOW$MY_MODEL_X(col1, col2, ..., colN)
```

### Limitations
* Has not been tested on Windows.
* Currently only supports `scikit-learn` and `xgboost` models.

## Development Setup
* Clone the repo locally.
* Install needed dev dependencies by `pip install -r dev_requirements.txt`
  * Recommend an fresh virtual environment with python 3.8.
* Intall package in local editable model for development by `pip install -e .`
* Run unit tests by `pytest tests/`

## Contributing
Please refer to [CONTRIBUTING.md](CONTRIBUTING.md).
