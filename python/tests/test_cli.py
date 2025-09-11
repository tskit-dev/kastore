import io
import logging
import os
import subprocess
import sys
import tempfile
from unittest import mock

import numpy as np
import pytest

import kastore as kas
import kastore.__main__ as main
import kastore.cli as cli

"""
Test cases for the kastore CLI.
"""


def capture_output(func, *args, **kwargs):
    """
    Runs the specified function and arguments, and returns the
    tuple (stdout, stderr) as strings.
    """
    stdout = sys.stdout
    sys.stdout = io.StringIO()
    stderr = sys.stderr
    sys.stderr = io.StringIO()

    try:
        func(*args, **kwargs)
        stdout_output = sys.stdout.getvalue()
        stderr_output = sys.stderr.getvalue()
    finally:
        sys.stdout.close()
        sys.stdout = stdout
        sys.stderr.close()
        sys.stderr = stderr
    return stdout_output, stderr_output


def test_cli_main():
    with mock.patch("argparse.ArgumentParser.parse_args") as mocked_parse:
        cli.kastore_main()
        mocked_parse.assert_called_once()


def test_main():
    with mock.patch("argparse.ArgumentParser.parse_args") as mocked_parse:
        main.main()
        mocked_parse.assert_called_once()


@pytest.fixture
def parser():
    return cli.get_kastore_parser()


def test_list_parser_defaults(parser):
    args = parser.parse_args(["ls", "filename"])
    assert args.store == "filename"
    assert args.long is False
    assert args.human_readable is False


@pytest.mark.parametrize("flag", ["-l", "--long"])
def test_list_parser_long(parser, flag):
    args = parser.parse_args(["ls", flag, "filename"])
    assert args.store == "filename"
    assert args.long is True


@pytest.mark.parametrize("flag", ["-H", "--human-readable"])
def test_list_parser_human_readable(parser, flag):
    args = parser.parse_args(["ls", flag, "filename"])
    assert args.store == "filename"
    assert args.human_readable is True


def test_dump_parser_defaults(parser):
    args = parser.parse_args(["dump", "filename", "array"])
    assert args.store == "filename"
    assert args.array == "array"


def run_command(cmd):
    stdout = subprocess.check_output(
        [sys.executable, "-m", "kastore"] + cmd, stderr=subprocess.STDOUT
    )
    return stdout.decode()


def test_help():
    stdout = run_command(["-h"])
    assert len(stdout) > 0


def test_version():
    stdout = run_command(["-V"])
    assert len(stdout) > 0
    assert stdout.split()[-1] == kas.__version__


@pytest.fixture
def cli_temp_file():
    fd, temp_file = tempfile.mkstemp(prefix="htsget_cli_test_")
    os.close(fd)
    yield temp_file
    os.unlink(temp_file)


def get_output(cmd):
    parser = cli.get_kastore_parser()
    args = parser.parse_args(cmd)
    return capture_output(args.runner, args)


def get_example_data():
    return {"A": np.arange(100), "B": np.zeros(10, dtype=int)}


def test_list_empty(cli_temp_file):
    kas.dump({}, cli_temp_file)
    stdout, stderr = get_output(["ls", cli_temp_file])
    assert len(stderr) == 0
    assert len(stdout) == 0
    for opts in ["-l", "-lH"]:
        stdout, stderr = get_output(["ls", opts, cli_temp_file])
        assert len(stderr) == 0
        assert len(stdout) == 0


def test_list(cli_temp_file):
    data = get_example_data()
    kas.dump(data, cli_temp_file)
    stdout, stderr = get_output(["ls", cli_temp_file])
    assert len(stderr) == 0
    assert stdout.splitlines() == sorted(list(data.keys()))


def test_list_long_format(cli_temp_file):
    data = get_example_data()
    kas.dump(data, cli_temp_file)
    stdout, stderr = get_output(["ls", cli_temp_file, "-l"])
    assert len(stderr) == 0
    lines = stdout.splitlines()
    assert len(lines) == len(data)
    # TODO weak test, check the actual output.


def test_list_long_format_human(cli_temp_file):
    data = get_example_data()
    kas.dump(data, cli_temp_file)
    stdout, stderr = get_output(["ls", cli_temp_file, "-l", "-H"])
    assert len(stderr) == 0
    lines = stdout.splitlines()
    assert len(lines) == len(data)
    # TODO weak test, check the actual output.


def test_dump(cli_temp_file):
    data = get_example_data()
    kas.dump(data, cli_temp_file)
    for key in data.keys():
        stdout, stderr = get_output(["dump", cli_temp_file, key])
        assert len(stderr) == 0
        assert stdout.splitlines() == list(map(str, data[key]))


def test_error(cli_temp_file):
    # We really should be catching this exception and writing a message to
    # stderr. For now though, document the fact that we are raising the exception
    # up to the main.
    with pytest.raises(EOFError):
        get_output(["ls", cli_temp_file])


def verify_logging(args, level, cli_temp_file):
    # We don't actually check the output here as we're mocking out the
    # call to logging config, but it's convenient to reuse the machinery
    # here in this function
    data = get_example_data()
    kas.dump(data, cli_temp_file)
    log_format = "%(asctime)s %(message)s"
    with mock.patch("logging.basicConfig") as mocked_config:
        stdout, stderr = get_output(args + ["ls", cli_temp_file])
        mocked_config.assert_called_once_with(level=level, format=log_format)
    return stderr


def test_logging_verbosity_0(cli_temp_file):
    verify_logging([], logging.WARNING, cli_temp_file)


def test_logging_verbosity_1(cli_temp_file):
    verify_logging(["-v"], logging.INFO, cli_temp_file)


def test_logging_verbosity_2(cli_temp_file):
    verify_logging(["-vv"], logging.DEBUG, cli_temp_file)
