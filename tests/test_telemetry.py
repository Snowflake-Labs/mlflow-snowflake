from typing import Any, Dict, Optional, Set

import pytest

from snowflake.ml.mlflow.util.telemetry import (
    TelemetryClient,
    TelemetryField,
    _usage_logging_helper,
)


class Recorder:
    """A recorder to record usage logging message content."""

    def record(self, fname, fparams, error, captured) -> None:
        self.fname = fname
        self.fparams = fparams
        self.error = error
        self.captured = captured

    def reset(self) -> None:
        self.fname = None
        self.fparams = None
        self.error = None
        self.captured = None


def record(fields: Set[str], recorder: Recorder, field_to_capture=None, capture_self=False):
    def _record(function_name: str, function_params: Dict[str, str], error: Optional[str], captured: Optional[Any]):
        recorder.record(function_name, function_params, error, captured)

    return _usage_logging_helper(
        fields=fields, handler=_record, field_to_capture=field_to_capture, capture_self=capture_self
    )


def test_usage_logging_helper_with_all_fields():
    """Test usage logging helper with all fields specified."""
    recorder = Recorder()

    @record(fields={"a", "b", "c"}, recorder=recorder)
    def target_func1(a, b=1, *, c=2):
        return a

    # Test with defaults.
    recorder.reset()
    r1 = target_func1(0)
    assert r1 == 0
    assert recorder.fname == "target_func1"
    assert recorder.error is None
    assert recorder.fparams["a"] == "0"
    assert recorder.fparams["b"] == "1"
    assert recorder.fparams["c"] == "2"
    assert recorder.captured is None
    # Test with explicit parameters
    recorder.reset()
    r2 = target_func1(99, 5, c=100)
    assert r2 == 99
    assert recorder.fname == "target_func1"
    assert recorder.error is None
    assert recorder.fparams["a"] == "99"
    assert recorder.fparams["b"] == "5"
    assert recorder.fparams["c"] == "100"
    assert recorder.captured is None


def test_usage_logging_helper_with_subset_fields():
    """Test usage logging helper with subset fields specified."""
    recorder = Recorder()

    @record(fields={"b", "c"}, recorder=recorder)
    def target_func1(a, b=1, *, c=2):
        pass

    # Test with defaults.
    recorder.reset()
    target_func1(0)
    assert recorder.fname == "target_func1"
    assert recorder.error is None
    assert "a" not in recorder.fparams
    assert recorder.fparams["b"] == "1"
    assert recorder.fparams["c"] == "2"
    # Test with explicit parameters
    recorder.reset()
    target_func1(99, 5, c=100)
    assert recorder.fname == "target_func1"
    assert recorder.error is None
    assert "a" not in recorder.fparams
    assert recorder.fparams["b"] == "5"
    assert recorder.fparams["c"] == "100"


def test_usage_logging_helper_with_variable_inputs():
    """Expect raise when input function has variable inputs."""
    recorder = Recorder()

    @record(fields={}, recorder=recorder)
    def target1(*args):
        pass

    with pytest.raises(ValueError, match="varargs and varkw are not supported."):
        target1()

    @record(fields={}, recorder=recorder)
    def target2(**kargs):
        pass

    with pytest.raises(ValueError, match="varargs and varkw are not supported."):
        target2()


def test_usage_logging_helper_with_missing_fields():
    """Expect raise when input function has missing fields."""
    recorder = Recorder()

    @record(fields={"a"}, recorder=recorder)
    def target1(b):
        pass

    @record(fields={"b", "c"}, recorder=recorder)
    def target2(b):
        pass

    with pytest.raises(ValueError, match=r".+are not found."):
        target1(3)
    with pytest.raises(ValueError, match=r".+are not found."):
        target2(3)


def test_usage_logging_helper_with_error():
    """Expect capture error message when input function raises."""

    class CustomException(Exception):
        pass

    recorder = Recorder()

    @record(fields={"a"}, recorder=recorder)
    def target1(a):
        raise CustomException("msg")

    with pytest.raises(CustomException, match="msg"):
        target1(3)
    assert recorder.fname == "target1"
    assert recorder.error == "CustomException('msg')"
    assert recorder.fparams["a"] == "3"
    assert recorder.captured is None


def test_telemetry_client(session):
    record = None

    def record_data(data):
        nonlocal record
        record = data

    session._conn._conn._telemetry.try_add_log_to_batch.side_effect = record_data
    client = TelemetryClient(session)
    client.send_usage("f1", {"a": "1"}, None)
    record_dict = record.to_dict()
    data = record_dict["message"][TelemetryField.KEY_DATA.value]
    assert data[TelemetryField.KEY_FUNC_NAME.value] == "f1"
    assert data[TelemetryField.KEY_FUNC_PARAMS.value] == {"a": "1"}
    assert TelemetryField.KEY_ERROR.value not in data
    assert record_dict["message"]["source"] == "SnowML"
    assert record_dict["message"]["type"] == "snowml_function_usage"
    assert record_dict["message"]["project"] == "MLOps"
    assert record_dict["message"]["subproject"] == "MLflowDeploymentPlugin"


def test_usage_logging_helper_with_field_to_captured_missing():
    """Expect raise when first arg is not available."""
    recorder = Recorder()

    @record(fields={}, recorder=recorder, field_to_capture="missing")
    def target1():
        pass

    with pytest.raises(ValueError, match=r"Failed to capture field.*"):
        target1()


def test_usage_logging_helper_with_field_to_captured_captured():
    recorder = Recorder()

    class Dummy:
        @record(fields={"a"}, recorder=recorder, field_to_capture="b")
        def target1(a, b=1):
            pass

    d = Dummy()
    d.target1()

    class Dummy2:
        @record(fields={"a"}, recorder=recorder, capture_self=True)
        def target1(a, b=1):
            pass

    d2 = Dummy2()
    d2.target1()
    assert recorder.captured == d2


def test_usage_logging_helper_with_with_conflict_cpature():
    """Expect to raise when more than one capture is specified."""
    recorder = Recorder()
    with pytest.raises(ValueError, match="Can not capture both field and self."):

        class Dummy:
            @record(fields={"a"}, recorder=recorder, field_to_capture="b", capture_self=True)
            def target1(a, b=1):
                pass
