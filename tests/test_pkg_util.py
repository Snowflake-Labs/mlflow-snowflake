from unittest.mock import MagicMock

import pytest

from snowflake.ml.mlflow.util.pkg_util import (
    DependencyException,
    _convert_to_snowflake_package_requirements,
    _parse_requirements,
    _rewritten_requirements_txt,
    _sanitize,
    check_compatibility_with_snowflake_conda_channel,
    extract_package_requirements,
    validate_conda_installation,
)

TEST_REQUIREMENTS = [
    "pkg1",
    "pkg2==1.1.1",
    "pkg3>=1.1.1,<=2.1.1",
    "pkg4>=2.1.1",
    "pkg5",
]


def test_sanitize():
    inputs = [
        "#",
        " pkg1 # inline",
        "pkg2 \\",
        "== \\",
        "1.1.1",
        "",
    ]
    res = _sanitize(inputs)
    expected = ["pkg1", "pkg2 == 1.1.1"]
    assert expected == res


def test_parse_requirements_when_invalid():
    """Expect raise for invalid requirement specs."""
    invalid_inputs = [
        "-c constraints.txt",
        "-r extra_requirements.txt",
        "pkg @ file://tmp",
        "pkg @ http://www.snowflake.com",
        "pkg[extra]>=1.1.1",
        "pkg[extra]>=1.1.1;python_version<'3.9'",
    ]
    for invalid_input in invalid_inputs:
        with pytest.raises(ValueError, match=r"Invalid requirement: .*"):
            _parse_requirements([invalid_input])


def test_parse_requirements():
    """Expect correct parsing for valid requirements."""
    res = _parse_requirements(TEST_REQUIREMENTS, exclusions={"pkg5"})
    # verify 'pkg5' is excluded.
    assert len(res) == 4
    assert res[0].parsed.name == "pkg1"
    assert res[1].pinned
    assert res[1].parsed.name == "pkg2"
    assert res[2].parsed.name == "pkg3"
    assert len(res[2].parsed.specs) == 2
    assert res[3].parsed.name == "pkg4"


def test_convert_to_snowflake_package_requirements_when_use_latest():
    """Expectations when use_latest is enabled."""
    parsed_requirements = _parse_requirements(TEST_REQUIREMENTS)
    res = _convert_to_snowflake_package_requirements(parsed_requirements, use_latest=True)
    expected = [
        "pkg1",
        "pkg2",
        "pkg3",
        "pkg4",
        "pkg5",
    ]
    assert res == expected


def test_convert_to_snowflake_package_requirements_when_not_use_latest():
    """Expectations when use_latest is disabled."""
    parsed_requirements = _parse_requirements(TEST_REQUIREMENTS)
    res = _convert_to_snowflake_package_requirements(parsed_requirements, use_latest=False)
    # pinned package versions will be respected
    expected = [
        "pkg1",
        "pkg2==1.1.1",
        "pkg3",
        "pkg4",
        "pkg5",
    ]
    assert res == expected


def test_rewritten_requirements_txt():
    """Expect rewriten `requirements.txt` has correct content."""
    reqs = _parse_requirements(TEST_REQUIREMENTS)
    with _rewritten_requirements_txt(reqs, use_latest=True) as f:
        res1 = open(f.name).read().splitlines()
        expected1 = [
            "pkg1",
            "pkg2",
            "pkg3",
            "pkg4",
            "pkg5",
        ]
        assert res1 == expected1

    with _rewritten_requirements_txt(reqs, use_latest=False) as f:
        res2 = open(f.name).read().splitlines()
        expected2 = [
            "pkg1",
            "pkg2==1.1.1",
            "pkg3>=1.1.1,<=2.1.1",
            "pkg4>=2.1.1",
            "pkg5",
        ]
        assert res2 == expected2


def test_extract_package_requirements_unmatched(
    mock_snow_channel,
):
    """Expect raise when requirements could not be satisfied."""
    mock_snow_channel.side_effect = ValueError("Package requirements could not be satisfied using Snowflake channel.")
    with pytest.raises(
        ValueError,
        match=r"Package requirements could not be satisfied using Snowflake channel.*",
    ):
        extract_package_requirements(TEST_REQUIREMENTS, exclusions={}, use_latest=True)


def test_extract_package_requirements(mock_snow_channel):
    res1 = extract_package_requirements(TEST_REQUIREMENTS, exclusions={"pkg5"}, use_latest=True)
    expected1 = [
        "pkg1",
        "pkg2",
        "pkg3",
        "pkg4",
    ]
    assert res1 == expected1
    res2 = extract_package_requirements(TEST_REQUIREMENTS, exclusions={}, use_latest=False)
    expected2 = [
        "pkg1",
        "pkg2==1.1.1",
        "pkg3",
        "pkg4",
        "pkg5",
    ]
    assert res2 == expected2


def test_validate_conda_installation_when_not_present(monkeypatch):
    """Expect raise when conda is not installed."""
    mock = MagicMock()
    monkeypatch.setattr("shutil.which", mock)
    mock.return_value = None
    with pytest.raises(RuntimeError, match=r"Could not find conda executable.*"):
        validate_conda_installation()


def test_validate_conda_installation_when_present(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr("shutil.which", mock)
    mock.return_value = "/opt/bin/miniconda"
    validate_conda_installation()


def test_check_compatibility_with_invalid_response(monkeypatch):
    """Expect raise when response is not invalid."""
    mock = MagicMock()
    monkeypatch.setattr("shutil.which", mock)
    mock.return_value = "/opt/bin/miniconda"
    mock2 = MagicMock()
    monkeypatch.setattr("subprocess.run", mock2)
    mock2.return_value.stdout = "invalid"
    with pytest.raises(DependencyException, match=r"Malformed response:.*"):
        check_compatibility_with_snowflake_conda_channel("path")


def test_check_compatibility_with_missing_packages(monkeypatch):
    """Expect raise when missing packages."""
    mock = MagicMock()
    monkeypatch.setattr("shutil.which", mock)
    mock.return_value = "/opt/bin/miniconda"
    mock2 = MagicMock()
    monkeypatch.setattr("subprocess.run", mock2)
    mock2.return_value.stdout = '{"packages":["pkg1","pkg2"],"exception_name":"PackagesNotFoundError"}'
    with pytest.raises(
        DependencyException, match=r"Package requirements could not be satisfied using Snowflake channel.*"
    ):
        check_compatibility_with_snowflake_conda_channel("path")
