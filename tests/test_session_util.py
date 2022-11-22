from unittest.mock import MagicMock

import pytest
from mlflow.exceptions import MlflowException

from snowflake.ml.mlflow import session_util as util


def test_basic_session_get_set(clean_session):
    """Test basic get and set operations for session util."""
    assert not util.is_session_set()
    util.set_session(clean_session)
    assert util.get_session() == clean_session
    assert util.is_session_set()


def test_create_session(clean_session):
    """Expect session created and set."""
    assert not util.is_session_set()
    util.create_session({"account": "2"})
    assert util.is_session_set()


def test_resolve_session_with_override(clean_session):
    """Expect to return explicitly set session."""
    assert not util.is_session_set()
    util.set_session(clean_session)
    session = util._resolve_session(target_uri="snowflake")
    assert session == clean_session


def test_resolve_session_with_no_connection(clean_session):
    """Expect return None without connection specified."""
    assert not util.is_session_set()
    session = util._resolve_session(target_uri="snowflake")
    assert not session


def test_resolve_session_with_connection_not_found(clean_session, monkeypatch):
    """Expect raise with connection specified not found."""
    m = MagicMock()
    monkeypatch.setattr("snowflake.ml.mlflow.util.snowsql_cfg_util.read_snowsql_cfgs", m)
    m.return_value = None
    assert not util.is_session_set()
    with pytest.raises(MlflowException, match=r"Failed to read SnowSQL configuration"):
        util._resolve_session(target_uri="snowflake:?connection=not_found")


def test_resolve_session_with_more_than_one_connection(clean_session):
    """Expect raise when more than one connection specified."""
    assert not util.is_session_set()
    with pytest.raises(MlflowException, match="Only a single connection name can be specified."):
        util._resolve_session(target_uri="snowflake:?connection=c1&connection=c2")


def test_resolve_session_with_connection(clean_session, monkeypatch):
    """Expect create session from configuration for specified connection."""
    assert not util.is_session_set()
    m = MagicMock(
        return_value={
            "account": "acct1",
            "user": "usr1",
            "password": "pass1",
        }
    )
    monkeypatch.setattr("snowflake.ml.mlflow.util.snowsql_cfg_util.read_snowsql_cfgs", m)
    session = util._resolve_session(target_uri="snowflake:?connection=connections.test")
    assert session
