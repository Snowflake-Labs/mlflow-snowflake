import numpy as np
import pandas as pd
import pytest
from mlflow.exceptions import MlflowException

from snowflake.ml.mlflow.deploy.udf_util import upload_model_from_mlflow


class TestUploadModelFromMLflow:
    def test_no_model_config(self, session, tmp_path):
        """Expect raise if no MLmodel file is present."""
        from snowflake.snowpark import Session

        with pytest.raises(MlflowException, match="Model config file did not exist."):
            upload_model_from_mlflow(Session(), model_dir_path=tmp_path, udf_name="nvm")

    def test_no_signature(self, session, tmp_path):
        """Expect raise if model signature is not present."""
        from mlflow.models import Model

        from snowflake.snowpark import Session

        config_path = tmp_path / "MLmodel"
        config_path.touch()
        Model().save(config_path)
        with pytest.raises(
            MlflowException,
            match="Signature is required to automatically generate the UDF.",
        ):
            upload_model_from_mlflow(Session(), model_dir_path=tmp_path, udf_name="nvm")

    def test_not_single_output(self, session, tmp_path):
        """Expect raise if model has more than one output."""
        from mlflow.models import Model
        from mlflow.models.signature import ModelSignature
        from mlflow.types.schema import ColSpec, DataType, Schema

        from snowflake.snowpark import Session

        outputs = Schema(inputs=[ColSpec(DataType.float, "f1"), ColSpec(DataType.float, "f2")])
        inputs = Schema(inputs=[ColSpec(DataType.float, "f0")])
        sig = ModelSignature(inputs=inputs, outputs=outputs)
        config_path = tmp_path / "MLmodel"
        config_path.touch()
        Model(signature=sig).save(config_path)
        with pytest.raises(MlflowException, match="Only single model output is supported."):
            upload_model_from_mlflow(Session(), model_dir_path=tmp_path, udf_name="nvm")

    def test_unsupported_flavors(self, session, tmp_path):
        """Expect raise if model flavor is not supported yet."""
        from mlflow.models import Model
        from mlflow.models.signature import ModelSignature
        from mlflow.types.schema import ColSpec, DataType, Schema

        from snowflake.snowpark import Session

        outputs = Schema(inputs=[ColSpec(DataType.float, "f1")])
        inputs = Schema(inputs=[ColSpec(DataType.float, "f0")])
        sig = ModelSignature(inputs=inputs, outputs=outputs)
        config_path = tmp_path / "MLmodel"
        config_path.touch()
        flavors = {"test": {"model_path": "/tmp/fun"}}
        Model(flavors=flavors, signature=sig).save(config_path)
        with pytest.raises(MlflowException, match=r".*flavors are not supported.*"):
            upload_model_from_mlflow(Session(), model_dir_path=tmp_path, udf_name="nvm")

    def test_no_flavors(self, session, tmp_path):
        """Expect raise if model flavor is not present."""
        from mlflow.models import Model
        from mlflow.models.signature import ModelSignature
        from mlflow.types.schema import ColSpec, DataType, Schema

        from snowflake.snowpark import Session

        outputs = Schema(inputs=[ColSpec(DataType.float, "f1")])
        inputs = Schema(inputs=[ColSpec(DataType.float, "f0")])
        sig = ModelSignature(inputs=inputs, outputs=outputs)
        config_path = tmp_path / "MLmodel"
        config_path.touch()
        Model(flavors=None, signature=sig).save(config_path)
        with pytest.raises(MlflowException, match="No flavor is specified for model deployment."):
            upload_model_from_mlflow(Session(), model_dir_path=tmp_path, udf_name="nvm")

    def test_test_data_dim_mismatch(self, session, tmp_path):
        """Expect raise if test data does not match in length."""
        from mlflow.models import Model
        from mlflow.models.signature import ModelSignature
        from mlflow.types.schema import ColSpec, DataType, Schema

        from snowflake.snowpark import Session

        outputs = Schema(inputs=[ColSpec(DataType.float, "f1")])
        inputs = Schema(inputs=[ColSpec(DataType.float, "f0")])
        sig = ModelSignature(inputs=inputs, outputs=outputs)
        config_path = tmp_path / "MLmodel"
        config_path.touch()
        flavors = {"sklearn": None}
        Model(flavors=flavors, signature=sig).save(config_path)
        with pytest.raises(MlflowException, match="Test data dimension mis-match."):
            upload_model_from_mlflow(
                Session(),
                model_dir_path=tmp_path,
                udf_name="nvm",
                test_data_X=pd.DataFrame(np.array([[1, 2], [3, 4]])),
                test_data_y=pd.Series(np.array([1])),
            )

    def test_test_data_missing(self, session, tmp_path):
        """Expect raise if test data is not fully specified."""
        from mlflow.models import Model
        from mlflow.models.signature import ModelSignature
        from mlflow.types.schema import ColSpec, DataType, Schema

        from snowflake.snowpark import Session

        outputs = Schema(inputs=[ColSpec(DataType.float, "f1")])
        inputs = Schema(inputs=[ColSpec(DataType.float, "f0")])
        sig = ModelSignature(inputs=inputs, outputs=outputs)
        config_path = tmp_path / "MLmodel"
        config_path.touch()
        flavors = {"sklearn": None}
        Model(flavors=flavors, signature=sig).save(config_path)
        with pytest.raises(
            MlflowException,
            match="Both input & output test data needs to be specified.",
        ):
            upload_model_from_mlflow(
                Session(),
                model_dir_path=tmp_path,
                udf_name="nvm",
                test_data_X=pd.DataFrame(np.array([[1, 2], [3, 4]])),
            )
