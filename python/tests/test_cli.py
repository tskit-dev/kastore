"""
Test cases for the kastore CLI.
"""
import io
import logging
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

import kastore as kas
import kastore.__main__ as main
import kastore.cli as cli


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


class TestMain(unittest.TestCase):
    """
    Simple tests for the main function.
    """

    def test_cli_main(self):
        with mock.patch("argparse.ArgumentParser.parse_args") as mocked_parse:
            cli.kastore_main()
            mocked_parse.assert_called_once()

    def test_main(self):
        with mock.patch("argparse.ArgumentParser.parse_args") as mocked_parse:
            main.main()
            mocked_parse.assert_called_once()


class TestListArgumentParser(unittest.TestCase):
    """
    Tests the parser to ensure it parses input values correctly.
    """

    def parse_args(self, args):
        parser = cli.get_kastore_parser()
        return parser.parse_args(args)

    def test_defaults(self):
        args = self.parse_args(["ls", "filename"])
        self.assertEqual(args.store, "filename")
        self.assertEqual(args.long, False)
        self.assertEqual(args.human_readable, False)

    def test_long(self):
        args = self.parse_args(["ls", "-l", "filename"])
        self.assertEqual(args.store, "filename")
        self.assertEqual(args.long, True)
        args = self.parse_args(["ls", "--long", "filename"])
        self.assertEqual(args.store, "filename")
        self.assertEqual(args.long, True)

    def test_human_readable(self):
        args = self.parse_args(["ls", "-H", "filename"])
        self.assertEqual(args.store, "filename")
        self.assertEqual(args.human_readable, True)
        args = self.parse_args(["ls", "--human-readable", "filename"])
        self.assertEqual(args.store, "filename")
        self.assertEqual(args.human_readable, True)


class TestDumpArgumentParser(unittest.TestCase):
    """
    Tests the parser to ensure it parses input values correctly.
    """

    def parse_args(self, args):
        parser = cli.get_kastore_parser()
        return parser.parse_args(args)

    def test_defaults(self):
        args = self.parse_args(["dump", "filename", "array"])
        self.assertEqual(args.store, "filename")
        self.assertEqual(args.array, "array")


class TestDirectOutput(unittest.TestCase):
    """
    Tests for some of the argparse inherited functionality
    """

    def run_command(self, cmd):
        stdout = subprocess.check_output(
            [sys.executable, "-m", "kastore"] + cmd, stderr=subprocess.STDOUT
        )
        return stdout.decode()

    def test_help(self):
        stdout = self.run_command(["-h"])
        self.assertGreater(len(stdout), 0)

    def test_version(self):
        stdout = self.run_command(["-V"])
        self.assertGreater(len(stdout), 0)
        self.assertEqual(stdout.split()[-1], kas.__version__)


class TestOutput(unittest.TestCase):
    """
    Tests that the output of the various tests is good.
    """

    def setUp(self):
        fd, self.temp_file = tempfile.mkstemp(prefix="htsget_cli_test_")
        os.close(fd)

    def tearDown(self):
        os.unlink(self.temp_file)

    def get_output(self, cmd):
        parser = cli.get_kastore_parser()
        args = parser.parse_args(cmd)
        return capture_output(args.runner, args)

    def get_example_data(self):
        data = {"A": np.arange(100), "B": np.zeros(10, dtype=int)}
        return data

    def test_list_empty(self):
        kas.dump({}, self.temp_file)
        stdout, stderr = self.get_output(["ls", self.temp_file])
        self.assertEqual(len(stderr), 0)
        self.assertEqual(len(stdout), 0)
        for opts in ["-l", "-lH"]:
            stdout, stderr = self.get_output(["ls", opts, self.temp_file])
            self.assertEqual(len(stderr), 0)
            self.assertEqual(len(stdout), 0)

    def test_list(self):
        data = self.get_example_data()
        kas.dump(data, self.temp_file)
        stdout, stderr = self.get_output(["ls", self.temp_file])
        self.assertEqual(len(stderr), 0)
        self.assertEqual(stdout.splitlines(), sorted(list(data.keys())))

    def test_list_long_format(self):
        data = self.get_example_data()
        kas.dump(data, self.temp_file)
        stdout, stderr = self.get_output(["ls", self.temp_file, "-l"])
        self.assertEqual(len(stderr), 0)
        lines = stdout.splitlines()
        self.assertEqual(len(lines), len(data))
        # TODO weak test, check the actual output.

    def test_list_long_format_human(self):
        data = self.get_example_data()
        kas.dump(data, self.temp_file)
        stdout, stderr = self.get_output(["ls", self.temp_file, "-l", "-H"])
        self.assertEqual(len(stderr), 0)
        lines = stdout.splitlines()
        self.assertEqual(len(lines), len(data))
        # TODO weak test, check the actual output.

    def test_dump(self):
        data = self.get_example_data()
        kas.dump(data, self.temp_file)
        for key in data.keys():
            stdout, stderr = self.get_output(["dump", self.temp_file, key])
            self.assertEqual(len(stderr), 0)
            self.assertEqual(stdout.splitlines(), list(map(str, data[key])))

    def test_error(self):
        # We really should be catching this exception and writing a message to
        # stderr. For now though, document the fact that we are raising the exception
        # up to the main.
        self.assertRaises(EOFError, self.get_output, ["ls", self.temp_file])

    def verify_logging(self, args, level):
        # We don't actually check the output here as we're mocking out the
        # call to logging config, but it's convenient to reuse the machinery
        # here in this class
        data = self.get_example_data()
        kas.dump(data, self.temp_file)
        log_format = "%(asctime)s %(message)s"
        with mock.patch("logging.basicConfig") as mocked_config:
            stdout, stderr = self.get_output(args + ["ls", self.temp_file])
            mocked_config.assert_called_once_with(level=level, format=log_format)
        return stderr

    def test_logging_verbosity_0(self):
        self.verify_logging([], logging.WARNING)

    def test_logging_verbosity_1(self):
        self.verify_logging(["-v"], logging.INFO)

    def test_logging_verbosity_2(self):
        self.verify_logging(["-vv"], logging.DEBUG)
