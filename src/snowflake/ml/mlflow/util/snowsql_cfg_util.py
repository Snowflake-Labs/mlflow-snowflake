import logging
import os
from os import path
from typing import Dict, List, Optional

from configobj import ConfigObj, ConfigObjError

_logger = logging.getLogger(__name__)

# Reference: https://docs.snowflake.com/en/user-guide/snowsql-config.html#snowsql-config-file
CFG_FILES: List[str] = [
    "/etc/snowsql.cnf",
    "/etc/snowflake/snowsql.cnf",
    "/usr/local/etc/snowsql.cnf",
    path.join(path.expanduser("~"), ".snowsql.cnf"),  # this is the original location
    path.join(path.expanduser("~"), ".snowsql", "config"),  # this is the new location
]

# Reference: https://docs.snowflake.com/en/user-guide/snowsql-start.html#connection-parameters-reference
SNOWSQL_PROTOCOL = "SNOWSQL_PROTOCOL"
SNOWSQL_ACCOUNT = "SNOWSQL_ACCOUNT"
SNOWSQL_DATABASE = "SNOWSQL_DATABASE"
SNOWSQL_SCHEMA = "SNOWSQL_SCHEMA"
SNOWSQL_WAREHOUSE = "SNOWSQL_WAREHOUSE"
SNOWSQL_ROLE = "SNOWSQL_ROLE"
SNOWSQL_HOST = "SNOWSQL_HOST"
SNOWSQL_PORT = "SNOWSQL_PORT"
SNOWSQL_PWD = "SNOWSQL_PWD"
SNOWSQL_USER = "SNOWSQL_USER"

_KEY_TO_ENV_VARS = {
    "account": SNOWSQL_ACCOUNT,
    "user": SNOWSQL_USER,
    "password": SNOWSQL_PWD,
    "host": SNOWSQL_DATABASE,
    "schema": SNOWSQL_SCHEMA,
    "role": SNOWSQL_ROLE,
    "port": SNOWSQL_PORT,
}

# Reference: https://docs.snowflake.com/en/user-guide/snowsql-start.html#configuring-default-connection-settings
DEFAULT_CONNECTION_NAME = "connections"

OAUTH_AUTHENTICATOR = "oauth"


def read_snowsql_cfgs(
    file_paths: List[str], connection_name: str = DEFAULT_CONNECTION_NAME
) -> Optional[Dict[str, str]]:
    """Load SnowSQL configurations.

    References:
        * https://docs.snowflake.com/en/user-guide/snowsql-start.html#using-named-connections

    Args:
        connection_name (str, optional): Name of the connection section in the config.
            Defaults to None.
        file_paths (List[str], optional): List of absolute file paths to the configurations.
            Later files will take precedence on values.
    Returns:
        Optional[Dict[str, str]]: Parsed SnowSQL configurations. None if specified connection
            does not exists.
    """
    cfg = ConfigObj()
    for file_path in file_paths:
        try:
            cfg.merge(ConfigObj(path.expanduser(file_path), interpolation=False))
        except ConfigObjError as e:
            _logger.error(f"Error parsing {file_path} Recovering partially " "parsed config values: cause: {e}")
            cfg.merge(e.config)
            pass

    if connection_name in cfg:
        return cfg[connection_name]
    return None


def extract_connection_params(cfg: Dict[str, str]) -> Dict[str, str]:
    """Extract connection paramters for Snowflake Python connector from SnowSQL
    configurations.

    # TODO(halu): Support OCSP, Proxy, keypair, MFA if requested.

    Reference:
        * https://docs.snowflake.com/en/user-guide/python-connector-api.html#connect
        as source of truth for all keys

    Args:
        cfg (Dict[str, str]): Parsed SnowSQL configurations.
    Raises:
        ValueError: If required fields are not specified.
    Returns:
        Dict[str, str]: Translated connection parameters for connector.
    """

    def _get(key: str):
        if key in cfg:
            return cfg[key]
        else:
            return None

    def _required(val: Optional[str], field_name: str):
        if not val:
            raise ValueError(f"{field_name} is required")
        return val

    # Required
    account = _required(_get("account") or _get("accountname"), "ACCOUNT")

    authenticator = _get("authenticator")
    token = _get("token")
    user = _get("user") or _get("username")
    password = _get("password")
    # OAuth requires `token`
    if authenticator and authenticator == OAUTH_AUTHENTICATOR:
        token = _required(token, "TOKEN")
    else:
        user = _required(user, "USER")
        password = _required(password, "PASSWORD")

    # Optional for the rest
    host = _get("host")
    port = _get("port")
    protocol = _get("protocol")
    database = _get("database") or _get("dbname")
    schema = _get("schema") or _get("schemaname")
    role = _get("role") or _get("rolename")
    warehouse = _get("warehouse") or _get("warehousename")
    # Keys have to be available for python-connector
    essentials = {
        "account": account,
        "user": user,
        "password": password,
        "host": host,
        "port": port,
        "protocol": protocol,
        "database": database,
        "schema": schema,
        "role": role,
        "warehouse": warehouse,
        "authenticator": authenticator,
        "token": token,
    }
    # Environment vars take precedence over config files.
    for (k, env_var) in _KEY_TO_ENV_VARS.items():
        v = os.environ.get(env_var)
        if v:
            essentials.update({k: v})
    # Filter out `None` values
    return {k: v for (k, v) in essentials.items() if v is not None}
