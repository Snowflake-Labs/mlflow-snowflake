# Snowflake MLflow Plugins

Currently, provides an experimental Snowflake Mlflow Deployment Plugin.
This enables Mlflow users to deploy external trained Mlflow packaged models to Snowflake.

This plugin implements the [Python API](https://www.mlflow.org/docs/latest/python_api/mlflow.deployments.html)
and [CLI](https://www.mlflow.org/docs/latest/cli.html#mlflow-deployments) for MLflow deployment plugins.

## Usage
### Prequisite
* The plugin relies on local anaconda installation to check if model dependencies could be satisfied. Suggest install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for minimum dependencies.
### Session connection
Two ways of connection are provided to establish a Snowflake session for deployment.
#### Python API
```python
from snowflake.mlflow import create_session
from mlflow.deployments import get_deploy_client
connection_parameters = dict()
create_session(connection_parameters)
target_uri = 'snowflake'
plugin = get_deploy_client(target_uri)
```
#### SnowSQL Configuration file
[SnowSQL Configuration file](https://docs.snowflake.com/en/user-guide/snowsql-config.html) is a familiar concept among existing SnowSQL CLI users and a neccessary way to establish connection to Snowflake if you intend to use MLflow CLI for deployment.
For the Snowflake deployment plugin the `target_uri` needs to have the`snowflake` scheme.
Connection parameters can be specified by adding `?connection={CONNECTION_NAME}`.
The `CONNECTION_NAME` references the connection specified in the SnowSQL configuration file e.g. `snowflake:/?connection=connections.ml`.
### Supported APIs
Following APIs are supported by both Python and CLI.
| Python API | CLI |
| ---| --- |
| `create_deployment`  | `mlflow deployments create`  |
| `delete_deployment`  | `mlflow deployments delete`  |
| `get_deployment` | `mlflow deployments get`  |
| `list_deployments` | `mlflow deployments list`  |

Detailed configuration options for `create_deployment` could be retrieved by  `mlflow deployments help -t snowflake`

### Limitations
* Has not been tested on Windows.
* Currently only supports `scikit-learn` and `xgboost` models

## Development Setup
* Clone the repo locally.
* Install needed dev dependencies by `pip install -r dev_requirements.txt`
  * Recommend an fresh virtual environment with python 3.8.
* Intall package in local editable model for development by `pip install -e .`
* Run unit tests by `pytest tests/`
