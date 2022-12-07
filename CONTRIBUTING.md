# Contributing to mlflow-snowflake
Hi, thank you for taking the time to improve Snowflake's MLflow Plugins!

## I have a feature request, or a bug report to submit.
We may already have existing bug reports and enhancement requests on our [issue tracker](https://github.com/Snowflake-Labs/mlflow-snowflake/issues).
Please start by checking these first!

## I've checked the [issue tracker](https://github.com/Snowflake-Labs/mlflow-snowflake/issues) and did not find an issue that matches my request.
In that case we'd love to hear from you!
Please [open a new issue](https://github.com/Snowflake-Labs/mlflow-snowflake/issues/new/choose) to get in touch with us.

## I'd like to contribute the bug fix or feature myself.
We encourage everyone to first [open a new issue](https://github.com/Snowflake-Labs/mlflow-snowflake/issues/new/choose) to discuss any feature work or bug fixes with one of the maintainers before starting any work.
In case you still want to start exploring, please follow the instructions below.

### Setup a development environment
#### Fork the repository and then clone the forked repo
```bash
git clone <YOUR_FORKED_REPO>
cd mlflow-snowflake
```
#### Install the library in edit mode and install dependencies
- Create a new Python virtual environment with any Python version that we support. Currently the only supported Python version is **Python 3.8**.
  ```bash
  conda create --name mlflow-snowflake-dev python=3.8
  ```
- Activate the new Python virtual environment.
  ```bash
  conda activate mlflow-snowflake-dev
  ```
- Go to the cloned repository root folder.
- Install dependencies
    ```bash
    pip install -r dev_requirements.txt
    ```
- Install the package in edit/development mode.
    ```bash
    python -m pip install -e .
    ```
  The `-e` tells `pip` to install the library in [edit, or development mode](https://pip.pypa.io/en/stable/cli/pip_install/#editable-installs).
## Tests
`pytest tests/` for unit testing.
