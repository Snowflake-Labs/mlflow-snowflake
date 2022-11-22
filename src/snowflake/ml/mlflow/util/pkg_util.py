import json
import subprocess
import tempfile
from contextlib import contextmanager
from itertools import filterfalse
from typing import Any, Iterable, List, NamedTuple, Optional, Set

import pkg_resources


def check_compatibility_with_snowflake_conda_channel(requirements_path: str) -> None:
    """Check if given `requirements.txt` could be satisfied against Snowflake conda channel.

    #TODO(halu): Handle `conda` path.

    Args:
        requirements_path (str): Absolute path to requirements.txt.
    """
    res = subprocess.run(
        [
            "conda",
            "create",
            "-c",
            "https://repo.anaconda.com/pkgs/snowflake",  # use Snowflake conda channel
            "-d",  # dry-run
            "--file",
            requirements_path,
            "--name",
            "test_for_compatibility",
            "--json",
            "--override-channels",
        ],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        jres = json.loads(res.stdout)
        pkgs = jres.get("packages")
        ex_type = jres.get("exception_name")
        if ex_type and ex_type == "PackagesNotFoundError" and pkgs:
            raise ValueError(f"Package requirements could not be satisfied using Snowflake channel.: {'|'.join(pkgs)}")
        else:
            raise ValueError("Package requirements could not be satisfied using Snowflake channel.")


class _Requirement(NamedTuple):
    # Parsed `Requirement`
    parsed: Any
    # Raw requirement line string
    raw: str
    # Whether requirement pin to specific version
    pinned: bool


def _is_comment(line: str) -> bool:
    return line.startswith("#")


def _is_empty(line: str) -> bool:
    return line == ""


def _strip_inline_comment(line: str) -> bool:
    return line[: line.find(" #")].rstrip() if " #" in line else line


def _join_continued_lines(lines: Iterable[str]) -> Iterable[str]:
    """Join all continued lines."""
    continued_lines = []

    for line in lines:
        if line.endswith("\\"):
            continued_lines.append(line.rstrip("\\"))
        else:
            continued_lines.append(line)
            yield "".join(continued_lines)
            continued_lines.clear()

    if continued_lines:
        yield "".join(continued_lines)


def _sanitize(requirements_lines: List[str]) -> List[str]:
    lines = map(str.strip, requirements_lines)
    lines = map(_strip_inline_comment, lines)
    lines = _join_continued_lines(lines)
    lines = filterfalse(_is_comment, lines)
    return list(filterfalse(_is_empty, lines))


def _parse_requirements(
    sanitized_requirements_lines: List[str],
    exclusions: Optional[Set[str]] = None,
) -> List[_Requirement]:
    """Parse `requirements.txt`.

    We intentionally support a subset of requirements.txt file specification.
    * DO NOT support `[[--option]...]`, e.g., -r extra_reqs.txt, -c constraints.txt.
    * DO NOT support `<archive url/path>`, all packages need to be available in channel.
    * DO NOT support marker and extras.

    Args:
        sanitized_requirements_lines (List[str]): Requirement lines.

    Returns:
        List[_Requirement]: List of parsed requirements.

    References:
        * PEP-508
         * Detailed requirement spec.
        * PEP-440
         * Detailed version spec.
        * https://pip.pypa.io/en/stable/reference/requirements-file-format/
         * file format which adds additional options on top of PE508.

    """
    res = []
    for raw_req in sanitized_requirements_lines:
        try:
            parsed_req = pkg_resources.Requirement.parse(raw_req)
        except Exception:
            raise ValueError(f"Invalid requirement: {raw_req}.")
        if parsed_req.marker or parsed_req.url or parsed_req.extras:
            raise ValueError(f"Invalid requirement: {raw_req}.")
        if exclusions and parsed_req.name in exclusions:
            continue
        pinned = False
        if parsed_req.specs and len(parsed_req.specs) == 1 and parsed_req.specs[0][0] == "==":
            pinned = True
        res.append(_Requirement(parsed=parsed_req, raw=raw_req, pinned=pinned))
    return res


def _convert_to_snowflake_package_requirements(reqs: List[_Requirement], use_latest=True) -> List[str]:
    """Convert to Snowflake package requirements.

    Snowflake currently only support two mode of package versions specifications:
        1) `pkg_name: This implies latest `pkg_name` version will be used.
        2) `pkg_name==ver`: This implies exact version `ver` will be used.

    Args:
        reqs (List[_Requirement]): List of parsed requirements.

    Returns:
        List[str]: Snowflake package requirements.
    """
    if use_latest:
        return [req.parsed.name for req in reqs]
    else:
        res = []
        for req in reqs:
            if req.pinned:
                res.append(f"{req.raw}")
            else:
                res.append(f"{req.parsed.name}")
        return res


@contextmanager
def _rewritten_requirements_txt(reqs: List[_Requirement], use_latest: bool):
    """A context wrapper for rewritten `requirements.txt` file.

    Args:
        reqs (List[_Requirement]): List of parsed requirements.
        use_latest (bool): Whether to use latest package.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt") as f:
        for r in reqs:
            if use_latest:
                f.write(f"{r.parsed.name}\n")
            else:
                f.write(f"{r.raw}\n")
        f.flush()
        yield f


def extract_package_requirements(requirements_lines, exclusions: Set[str] = None, use_latest=True) -> List[str]:
    """Extract package requirements for Snowflake Python UDF.

    Details of `requirements.txt` parsing support can be found in docstring of `_parse_requirements`.

    Args:
        requirements_lines (List[str]): Requirement lines.
        exclusions (dict, optional): Packages to be excluded from requirements. Defaults to {'mlflow'}.
        use_latest (bool, optional): Whether to use latest package versions avaialble. Defaults to True.

    Returns:
        List[str]: Package requirements that could be fed into Snowflake python packages specification directly.
    """
    sanitized_req_lines = _sanitize(requirements_lines)
    reqs = _parse_requirements(sanitized_req_lines, exclusions)
    with _rewritten_requirements_txt(reqs, use_latest) as f:
        check_compatibility_with_snowflake_conda_channel(f.name)
        return _convert_to_snowflake_package_requirements(reqs, use_latest)
