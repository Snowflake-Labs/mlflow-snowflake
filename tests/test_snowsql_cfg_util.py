import pytest
from configobj import ConfigObj

from snowflake.ml.mlflow.util.snowsql_cfg_util import (
    DEFAULT_CONNECTION_NAME,
    extract_connection_params,
    read_snowsql_cfgs,
)

CONNECTION_T1 = "connections.t1"
CONNECTION_T2 = "connections.t2"


@pytest.fixture
def snowsql_cfg_path_0(tmpdir):
    p = tmpdir.mkdir("config0").join("snowsql.cnf")
    config = ConfigObj()
    config.filename = str(p)
    config[DEFAULT_CONNECTION_NAME] = {
        "account": "snowflake1",
        "user": "admin",
        "password": "test",
        "host": "snowflake",
        "port": "8082",
    }
    config[CONNECTION_T1] = {
        "account": "snowflake2",
        "user": "admin",
        "password": "test",
        "host": "snowflake",
        "port": "8082",
    }
    config.write()
    return str(p)


@pytest.fixture
def snowsql_cfg_path_1(tmpdir):
    p = tmpdir.mkdir("config1").join("snowsql.cnf")
    config = ConfigObj()
    config.filename = str(p)
    config[DEFAULT_CONNECTION_NAME] = {
        "account": "snowflake3",
        "user": "admin",
        "password": "test",
        "host": "snowflake",
        "port": "8082",
    }
    config[CONNECTION_T1] = {
        "account": "snowflake4",
        "user": "admin",
        "password": "test",
        "host": "snowflake",
        "port": "8082",
    }
    config[CONNECTION_T2] = {
        "account": "snowflake5",
        "user": "admin",
        "password": "test",
        "host": "snowflake",
        "port": "8082",
    }
    config.write()
    return str(p)


def test_read_snowsql_cfgs_without_connection_filter(snowsql_cfg_path_0):
    """Expect to read default connection."""
    res = read_snowsql_cfgs([snowsql_cfg_path_0])
    assert res["account"] == "snowflake1"


def test_read_snowsql_cfgs_with_connection_filter(snowsql_cfg_path_0):
    """Expect to read only the specified named connection."""
    res = read_snowsql_cfgs([snowsql_cfg_path_0], connection_name=CONNECTION_T1)
    assert res["account"] == "snowflake2"
    res1 = read_snowsql_cfgs([snowsql_cfg_path_0], connection_name="connections.rand")
    assert not res1


def test_read_snowsql_later_cfg_take_precedence(snowsql_cfg_path_0, snowsql_cfg_path_1):
    """Expect values in later files take precedance."""
    res = read_snowsql_cfgs([snowsql_cfg_path_0, snowsql_cfg_path_1])
    # default take value from later configuration.
    assert res["account"] == "snowflake3"
    res1 = read_snowsql_cfgs([snowsql_cfg_path_0, snowsql_cfg_path_1], connection_name=CONNECTION_T1)
    assert res1["account"] == "snowflake4"
    res2 = read_snowsql_cfgs([snowsql_cfg_path_0, snowsql_cfg_path_1], connection_name=CONNECTION_T2)
    assert res2["account"] == "snowflake5"


@pytest.fixture
def parsed_snowsql_cfg():
    return {
        "accountname": "acct1",
        "username": "usr1",
        "password": "pwd1",
        "host": "snowflake.qa1",
        "port": "123",
        "protocol": "http",
        "dbname": "db",
        "schemaname": "schema",
        "rolename": "role",
        "warehousename": "warehouse",
        "authenticator": "default",
        "token": "path",
    }


MIN_SNOWSQL_CFG = {
    "account": "acct1",
    "user": "usr1",
    "password": "pwd1",
}

OAUTH_SNOWSQL_CFG = {
    "account": "acct2",
    "token": "token2",
    "authenticator": "oauth",
}

OPTIONAL_SNOWSQL_CFG = {
    "host": "snowflake.qa1",
    "port": "123",
    "protocol": "http",
    "database": "db",
    "schema": "schema",
    "role": "role",
    "warehouse": "warehouse",
    "authenticator": "default",
    "token": "path",
}

EXPECTED_SNOWSQL_CFG = {**MIN_SNOWSQL_CFG, **OPTIONAL_SNOWSQL_CFG}


def test_extract_connection_params_missing_required():
    """Expect raise when required params are missing."""
    with pytest.raises(ValueError):
        extract_connection_params({"accountname": "the"})
    with pytest.raises(ValueError):
        extract_connection_params({"authenticator": "oauth", "account": "acct"})


def test_extract_connection_params_with_oauth():
    """Expect pass with all OAuth fields available."""
    res = extract_connection_params(OAUTH_SNOWSQL_CFG)
    assert res["token"] == "token2"


def test_extract_connection_params_with_all_fields(parsed_snowsql_cfg):
    """Expect mapping from SnowSQL key to Python connector key."""
    res = extract_connection_params(parsed_snowsql_cfg)
    assert res == EXPECTED_SNOWSQL_CFG


def test_extract_connection_params_with_none_filtered():
    """Expect `None` value fields are filtered."""
    res = extract_connection_params(MIN_SNOWSQL_CFG)
    assert "warehouse" not in res


def test_extract_connection_params_with_env_var(parsed_snowsql_cfg, monkeypatch):
    """Expect environment variables take precedences over configurations."""
    monkeypatch.setenv("SNOWSQL_ACCOUNT", "acct2")
    res = extract_connection_params(parsed_snowsql_cfg)
    assert res["account"] == "acct2"
