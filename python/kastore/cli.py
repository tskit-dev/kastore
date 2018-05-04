"""
The CLI for kastore. Provides utilities for examining kastore files.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import argparse
import logging
import os
import signal
import sys

# import kastore


logger = logging.getLogger(__name__)


def error_message(message):
    """
    Writes an error message to stderr.
    """
    print("{}: error: {}".format(sys.argv[0], message), file=sys.stderr)


def run(args):
    log_level = logging.WARNING
    if args.verbose == 1:
        log_level = logging.INFO
    elif args.verbose >= 2:
        log_level = logging.DEBUG
    logging.basicConfig(format='%(asctime)s %(message)s', level=log_level)

    print("RUN:", args)


def get_kastore_parser():
    parser = argparse.ArgumentParser(
        description="Command line utility for kastore files.")
    parser.add_argument(
        "-V", "--version", action='version',
        # version='%(prog)s {}'.format(kastore.__version__))
        version='%(prog)s {}'.format(0))
    parser.add_argument(
        '--verbose', '-v', action='count', default=0,
        help="Increase verbosity.")

    return parser


def kastore_main():
    if os.name == "posix":
        # Set signal handler for SIGPIPE to quietly kill the program.
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    parser = get_kastore_parser()
    args = parser.parse_args()
    run(args)
