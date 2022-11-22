from typing import List, Optional, Tuple

from mlflow.exceptions import MlflowException

from snowflake.snowpark import Session


class DeploymentHelper:
    """Provide naive UDF deployment management.

    TODO(halu): Current implementation is based on existing primitive UDF management
    tooling. Will migrate once a more mature deployment management story is designed.
    """

    _MLFLOW_MODEL_DESCRIPTION = "from_snowflake_mlflow_plugin"

    def __init__(self, session: Session) -> None:
        self._session = session

    def tag_deployment(self, name: str) -> None:
        """Tag the UDF deployment to be tracked by the deployment plugin overriding the comment.

        Args:
            session (Session): A snowpark session.
            name (str): Name for the deployment.

        Raises:
            MlflowException: If fail to update `comment` for the deployment.
            MlflowException: If the deployment does not exist.
        """
        signature = self._get_deployment(name)
        res = self._session.sql(
            f"ALTER FUNCTION IF EXISTS {signature} SET COMMENT = '{self._MLFLOW_MODEL_DESCRIPTION}'"
        ).collect()
        if len(res) != 1 or res[0].status != "Statement executed successfully.":
            raise MlflowException(
                f"Failed to update description for deployment {signature}. "
                "Plugin will not be able to track this deployment. Please retry."
            )

    def _get_deployment(self, name: str, target_description: Optional[str] = None) -> str:
        """Get the deployment.

        Args:
            name (str): Name for the deployment.
            target_description (Optional[str], optional): Target description to match.
                Defaults to None.

        Raises:
            MlflowException: If there is more than one matching deployment.
            MlflowException: If no deployment matching `target_description` exists.

        Returns:
            str: Signature for the deployment.
        """
        res = self._session.sql(f"SHOW USER FUNCTIONS LIKE '%{name}%'").collect()
        if len(res) == 0:
            raise MlflowException(f"Deployment {name} does not exist.")
        if len(res) > 1:
            raise MlflowException(f"More than one deployment with name {name} exist.")
        if target_description and res[0].description != target_description:
            raise MlflowException(f"Deployment {name} does not exist.")
        return self._function_signature(res[0].arguments)

    def get_deployment(self, name: str) -> str:
        """Get deployment managed by this deployment plugin.

        Args:
            name (str): Name for the deployment.

        Raises:
            MlflowException: If there is more than one matching deployment.
            MlflowException: If the deployment does not exist.

        Returns:
            str: Signature for the deployment.
        """
        return self._get_deployment(name, self._MLFLOW_MODEL_DESCRIPTION)

    def list_deployments(self) -> List[Tuple[str, str]]:
        """List all the deployments managed by the deployment plugin.

        Returns
            List[Tuple[str, str]]: (Name, Function signature) of the deployments.
        """
        res = []
        qres = self._session.sql("SHOW USER FUNCTIONS").collect()
        for func in qres:
            if func.description == self._MLFLOW_MODEL_DESCRIPTION:
                res.append((func.name, self._function_signature(func.arguments)))
        return res

    def delete_deployment(self, name) -> None:
        """Delete the deployment."""
        try:
            signature = self.get_deployment(name)
        except Exception:
            return
        self._session.sql(f"DROP FUNCTION IF EXISTS {signature}").collect()

    @staticmethod
    def _function_signature(argument: str) -> str:
        """Extract function signature from function arguments string.

        Args:
            argument (str): Function argument in form of
                `name(type1, type2..) RETURN typeN`

        Returns:
            str: Signature for the UDF function.
        """
        return argument[: argument.find(" RETURN")].rstrip()
