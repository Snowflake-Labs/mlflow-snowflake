# CHANGELOG
## [Unreleased]
### Added
### Changed
### Fixed

## 0.0.1 (2022-12-01)
Initial release supporting deploying MLflow packaged SKLearn and XGBoost models to Snowflake.
### Added
### Changed
### Fixed


## 0.0.2 (2022-02-13)
Release to address some early PrPr feedbacks.
### Added
 * Support any model flavors as long as corresponding ML package is available in Snowflake conda channel.
 * Add return for `create_deployment`.
 * Avoid hard conda dependencies. Client does not require to run under conda env anymore.
 * Added a simple e2e python notebook.
### Changed
### Fixed


## 0.1.0 (2022-02-28)
Merge PR contributions from community and update telemetry message.
### Added
 * [18](https://github.com/Snowflake-Labs/mlflow-snowflake/pull/18) Better error message for incompatible data types.
 * [20](https://github.com/Snowflake-Labs/mlflow-snowflake/pull/20) Allow `predict` to return Snowpark DataFrame.
### Changed
 * Update telemetry message.
### Fixed
