from typing import Dict, List, Optional, Union
from urllib.parse import parse_qs, urlparse

from mlflow.exceptions import MlflowException

from snowflake.ml.mlflow.util import snowsql_cfg_util
from snowflake.snowpark import Session

_SESSION = None

QUERY_KEY_CONNECTION = "connection"


def is_session_set() -> bool:
    """Check if the Snowpark session has been set.

    Returns:
        bool: True if already set.
    """
    return _SESSION is not None


def set_session(session: Session) -> None:
    """Set the session for the Snowflake MLflow Plugin.

    Args:
        session (Session): A Snowpark session.
    """
    global _SESSION
    _SESSION = session


def get_session() -> Optional[Session]:
    """Get the current session for the Snowflake MLflow Plugin.

    Returns:
        Optional[Session]: A snowpark session.
    """
    global _SESSION
    return _SESSION


def create_session(options: Dict[str, Union[int, str]]) -> Session:
    """Establishes a Snowpark session from connection parameters.

    Refer to https://docs.snowflake.com/en/user-guide/python-connector-api.html#connect
    for detail options for connection parameters.

    Args:
        options (Dict[str, Union[int, str]]): Connection parameters.
    Returns:
        Session: A Snowpark session.
    """
    session = Session.builder.configs(options).create()
    set_session(session)
    return session


def _resolve_session(target_uri: str, cfg_file_paths: List[str] = snowsql_cfg_util.CFG_FILES) -> Optional[Session]:
    """For internal usage to resolve the Snowpark session.

    The global session will be used if set.
    If the global session is not set, we will attempt to create a session
    using the SnowSQL configuration locations and update the global session.

    Args:
        target_uri (str): URI for Snowflake MLflow deployment plugin.
        cfg_file_paths (List[str], optional): SnowSQL configuration file paths.
            Defaults to snowsql_cfg_util.CFG_FILES.

    Raises:
        MlflowException: Failure to find the specified connection from configuration files.
        MlflowException: If more than one connection is specified in `target_uri`.

    Returns:
        Optional[Session]: A Snowpark session.
    """
    session = get_session()
    if session:
        return session
    # Fallback to `target_uri`
    parsed = urlparse(target_uri)
    qss = parse_qs(parsed.query)
    if not qss or QUERY_KEY_CONNECTION not in qss:
        return None
    if len(qss[QUERY_KEY_CONNECTION]) == 1:
        connection_name = qss[QUERY_KEY_CONNECTION][0]
        cfg = snowsql_cfg_util.read_snowsql_cfgs(cfg_file_paths, connection_name=connection_name)
        if not cfg:
            raise MlflowException(f"Failed to read SnowSQL configuration for connection {connection_name}.")
        conn_params = snowsql_cfg_util.extract_connection_params(cfg)
        return create_session(conn_params)
    else:
        raise MlflowException("Only a single connection name can be specified.")


def _clear_session():
    """Only for testing."""
    global _SESSION
    _SESSION = None
