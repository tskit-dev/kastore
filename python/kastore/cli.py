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

import humanize

import kastore


logger = logging.getLogger(__name__)


def setup_logging(args):
    log_level = logging.WARNING
    if args.verbose == 1:
        log_level = logging.INFO
    elif args.verbose >= 2:
        log_level = logging.DEBUG
    logging.basicConfig(format='%(asctime)s %(message)s', level=log_level)


def _list(store, args):
    keys = sorted(store.keys())
    if args.long:
        sizes = []
        dtypes = []
        lengths = []
        for key in keys:
            info = store.info(key)
            dtypes.append(info.dtype)
            lengths.append(str(info.shape[0]))
            size = str(info.size)
            if args.human_readable:
                size = humanize.naturalsize(info.size, gnu=True)
            sizes.append(size)
        # Compute the column sizes
        dtype_colsize = max(map(len, dtypes))
        length_colsize = max(map(len, lengths))
        size_colsize = max(map(len, sizes))
        for key, dtype, length, size in zip(keys, dtypes, lengths, sizes):
            row = "{:<{}} {:>{}} {:>{}} {}".format(
                dtype, dtype_colsize,
                length, length_colsize,
                size, size_colsize, key)
            print(row)
    else:
        for key in keys:
            print(key)


def _dump(store, args):
    array = store[args.array]
    for value in array:
        s = "{}".format(value)
        # We should just print this out directly but it makes testing the output
        # in Python2 very awkward.
        print(s)


def run_list(args):
    setup_logging(args)
    with kastore.load(args.store) as store:
        _list(store, args)


def run_dump(args):
    setup_logging(args)
    with kastore.load(args.store) as store:
        _dump(store, args)


def add_store_argument(parser):
    parser.add_argument("store", help="The input kastore file")


def add_array_argument(parser):
    parser.add_argument("array", help="The name of the array of interest.")


def get_kastore_parser():
    top_parser = argparse.ArgumentParser(
        description="Command line utility for kastore files.")
    top_parser.add_argument(
        "-V", "--version", action='version',
        version='%(prog)s {}'.format(kastore.__version__))
    top_parser.add_argument(
        '--verbose', '-v', action='count', default=0,
        help="Increase verbosity.")
    # This is required to get uniform behaviour in Python2 and Python3
    subparsers = top_parser.add_subparsers(dest="subcommand")
    subparsers.required = True

    parser = subparsers.add_parser(
        "ls",
        help="List the contents of a store")
    add_store_argument(parser)
    parser.add_argument(
        "--long", "-l", action="store_true",
        help="Show details about arrays")
    parser.add_argument(
        "--human-readable", "-H", action="store_true",
        help="Show array sizes in human-readable format.")
    parser.set_defaults(runner=run_list)

    parser = subparsers.add_parser(
        "dump",
        help="Dump an array within the file to stdout")
    add_store_argument(parser)
    add_array_argument(parser)
    parser.set_defaults(runner=run_dump)

    return top_parser


def kastore_main():
    if os.name == "posix":
        # Set signal handler for SIGPIPE to quietly kill the program.
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    parser = get_kastore_parser()
    args = parser.parse_args()
    args.runner(args)
