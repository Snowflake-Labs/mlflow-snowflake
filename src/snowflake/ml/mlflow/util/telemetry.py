import functools
import inspect
import platform
import time
from enum import Enum, unique
from typing import Any, Callable, Dict, Optional, Set, Tuple

from snowflake.connector.telemetry import (
    TelemetryClient as PCTelemetryClient,
    TelemetryData as PCTelemetryData,
    TelemetryField as PCTelemetryField,
)
from snowflake.ml.mlflow import __version__ as mlflow_plugin_version
from snowflake.snowpark import Session


def _get_time_millis() -> int:
    """Returns the current time in milliseconds."""
    return int(time.time() * 1000)


@unique
class TelemetryField(Enum):
    # Top level message keys for telemetry
    KEY_PYTHON_VERSION = "python_version"
    KEY_OS = "operating_system"
    KEY_DATA = "data"
    KEY_VERSION = "version"
    # Message keys for `data`
    KEY_FUNC_NAME = "func_name"
    KEY_FUNC_PARAMS = "func_params"
    KEY_ERROR_INFO = "error_info"
    KEY_PROJECT = "project"
    KEY_SUBPROJECT = "subproject"
    KEY_CATEGORY = "category"


class TelemetryClient:
    """A in-band telemetry client leveraging the underlying Snowflake python connector."""

    _APPLICATION_NAME: str = "MLflowDeploymentPlugin"
    _PYTHON_VERSION = platform.python_version()
    _OS_NAME = platform.system()
    _CLIENT_VERSION = mlflow_plugin_version

    def __init__(self, session: Session) -> None:
        self._client: PCTelemetryClient = session._conn._conn._telemetry

    def _get_basic_data(self) -> Dict[str, str]:
        message = {
            PCTelemetryField.KEY_SOURCE.value: "SnowML",
            TelemetryField.KEY_VERSION.value: self._CLIENT_VERSION,
            TelemetryField.KEY_PYTHON_VERSION.value: self._PYTHON_VERSION,
            TelemetryField.KEY_OS.value: self._OS_NAME,
            PCTelemetryField.KEY_TYPE.value: "snowml_function_usage",
            TelemetryField.KEY_PROJECT.value: "MLOps",
            TelemetryField.KEY_SUBPROJECT.value: self._APPLICATION_NAME,
        }
        return message

    def send(self, msg: Dict[str, Any], timestamp: Optional[int] = None) -> None:
        """Send in-band client loggging.

        Args:
            msg (Dict[str, Any]): Logging message.
            timestamp (Optional[int], optional): Client timestamp for the message. Defaults to None.
        """
        if not timestamp:
            timestamp = _get_time_millis()
        telemetry_data = PCTelemetryData(message=msg, timestamp=timestamp)
        self._client.try_add_log_to_batch(telemetry_data)

    def send_usage(self, function_name: str, function_params: Dict[str, str], error: Optional[str] = None) -> None:
        """Send in-band function usage client logging.

        Args:
            function_name (str): Function name.
            function_params (Dict[str, str]): Function parameters.
            error (Optional[str], optional): Error. Defaults to None.
        """
        msg = self._get_basic_data()
        message = {
            **msg,
            TelemetryField.KEY_DATA.value: {
                TelemetryField.KEY_FUNC_NAME.value: function_name,
                TelemetryField.KEY_CATEGORY.value: "usage",
                TelemetryField.KEY_FUNC_PARAMS.value: {**function_params},
            },
        }
        if error:
            message = {
                **message,
                TelemetryField.KEY_ERROR_INFO.value: error,
            }
        self.send(message)
        # Switch to flush per X second when there's increased usage.
        self.flush()

    def flush(self):
        self._client.send_batch()


def _extract_arg_value(field: str, func_spec, args, kwargs) -> Tuple[bool, Any]:
    """Function to extract an specified argument value.
    Args:
        field (str): Target function argument name to extract.
        func_spec (FullArgSpec): Full argument spec for the function.
        args: `args` for the invoked function.
        kwargs: `kwargs` for the invoked function.

    Returns:
        Tuple[bool, Any]: First value indicates if `field` exists.
            Second value is the extracted value if existed.
    """
    if field in func_spec.args:
        idx = func_spec.args.index(field)
        if idx < len(args):
            return (True, args[idx])
        elif field in kwargs:
            return (True, kwargs[field])
        else:
            required_len = len(func_spec.args) - len(func_spec.defaults)
            return (True, func_spec.defaults[idx - required_len])
    elif field in func_spec.kwonlyargs:
        if field in kwargs:
            return (True, kwargs[field])
        else:
            return (True, func_spec.kwonlydefaults[field])
    else:
        return (False, None)


def _usage_logging_helper(
    *,
    fields: Set[str],
    field_to_capture: Optional[str] = None,
    capture_self: bool = False,
    handler: Callable[[str, Dict[str, str], Optional[str], Optional[Any]], None],
):
    """An utility to construct an function decorator for function usage logging.

    Examples:
        def usage_logging(fields):
            handler = ...
            return usage_logging_helper(fields=fields, handler=handler)

        @usage_loggging(fields={'a'})
        def target(a):
            pass

    Args:
        fields (Set[str]): Set of function parameter fields to record.
        field_to_capture (str, optional): Field value to capture to pass to `handler` as last arguement.
            Useful to capture `self`.
        capture_self(bool, optional): Whether to capture self for an instance method as `captured`.
        handler (Callable[[str, Dict[str, str], Optional[str], Any], None]):
            A handler function to handle function name, function parameters and
            error raised by the decorated function. Captured object would be last
            argument if requested.
            Example:
                def handler(function_name: str, function_params: Dict[str, str],
                    error: Optional[str], captured: Optional[Any]):
                    # `function_name` and `function_params` are from decorated function.
                    client = construct_client_from_captured(captured)
                    client.log(function_name, function_params, error)


    Returns:
        A function decorator.
    """
    if field_to_capture is not None and capture_self:
        raise ValueError("Can not capture both field and self.")

    def usage_logging_wrap(func):
        @functools.wraps(func)
        def wrap(*args, **kwargs):
            try:
                params = dict()
                spec = inspect.getfullargspec(func)
                if spec.varargs or spec.varkw:
                    raise ValueError("varargs and varkw are not supported.")
                not_found = set()
                for field in fields:
                    found, extracted_value = _extract_arg_value(field, spec, args, kwargs)
                    if not found:
                        not_found.add(field)
                    else:
                        params[field] = repr(extracted_value)
                if not_found:
                    raise ValueError(f"Fields {not_found} are not found.")
                error = None
                return func(*args, **kwargs)
            except Exception as e:
                error = repr(e)
                raise
            finally:
                captured = None
                if field_to_capture is not None:
                    field_found, captured = _extract_arg_value(field_to_capture, spec, args, kwargs)
                    if not field_found:
                        raise ValueError(f"Failed to capture field {field_to_capture}")
                elif capture_self:
                    captured = args[0]
                handler(func.__name__, params, error, captured)

        return wrap

    return usage_logging_wrap


def method_session_usage_logging_helper(
    sesssion_extractor: Callable[[Any], Session],
    fields_to_log: Set[str],
):
    """An utility function to generate sessiong logging decorator for classes.

    Examples:
        class Target:
            def usage_logging(fields_to_log):
                def session_extractor(capture):
                    return capture.get_session()
                return method_session_usage_logging_helper(
                    fields_to_log=fields_to_log,
                    sesssion_extractor=session_extractor,
                )

            @usage_loggging(fields_to_log={'a'})
            def target(a):
                pass

    Args:
        sesssion_extractor (Callable[[Any], Session]): A function to extract `Session` object
            from `self` instance.
        fields_to_log (Set[str]): Set[str].

    Returns:
        A function decorator.
    """

    def handler(function_name: str, function_params: Dict[str, str], error: Optional[str], captured: Optional[Any]):
        assert captured is not None
        session = sesssion_extractor(captured)
        client = TelemetryClient(session)
        client.send_usage(function_name, function_params, error)

    return _usage_logging_helper(fields=fields_to_log, capture_self=True, handler=handler)
