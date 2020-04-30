"""
Tests for load()ing and dump()ing with file-like objects.
"""
import multiprocessing
import os
import pathlib
import platform
import queue
import shutil
import socket
import socketserver
import tempfile
import traceback
import unittest

import numpy as np

import kastore as kas


IS_WINDOWS = platform.system() == "Windows"


class DictVerifyMixin:
    def verify_dicts_equal(self, d1, d2):
        self.assertEqual(sorted(d1.keys()), sorted(d2.keys()))
        for key in d1.keys():
            np.testing.assert_equal(d1[key], d2[key])


# pathlib.Path objects should work  transparently, and these tests check it
# isn't broken by fileobj-enabling code.
class PathlibMixin(DictVerifyMixin):
    def setUp(self):
        fd, path = tempfile.mkstemp(prefix="kas_test_pathlib")
        os.close(fd)
        self.temp_file = pathlib.Path(path)

    def tearDown(self):
        self.temp_file.unlink()

    def test_dump_to_pathlib_Path(self):
        data = {"a": np.arange(10)}
        kas.dump(data, self.temp_file, engine=self.engine)

    def test_load_from_pathlib_Path(self):
        data = {"a": np.arange(10)}
        kas.dump(data, str(self.temp_file), engine=self.engine)
        file_size = self.temp_file.stat().st_size
        for read_all in [True, False]:
            data_out = kas.load(self.temp_file, read_all=read_all, engine=self.engine)
            data2 = dict(data_out.items())
            file_size2 = self.temp_file.stat().st_size
            self.verify_dicts_equal(data, data2)
            self.assertEqual(file_size, file_size2)


class FileobjMixin(DictVerifyMixin):
    def setUp(self):
        fd, path = tempfile.mkstemp(prefix="kas_test_fileobj")
        os.close(fd)
        self.temp_file = path

    def tearDown(self):
        os.unlink(self.temp_file)

    def test_dump_fileobj_single(self):
        data = {"a": np.arange(10)}
        with open(self.temp_file, "wb") as f:
            kas.dump(data, f, engine=self.engine)
        data_out = kas.load(self.temp_file, engine=self.engine)
        data2 = dict(data_out.items())
        self.verify_dicts_equal(data, data2)

    def test_dump_fileobj_multi(self):
        with open(self.temp_file, "wb") as f:
            for i in range(10):
                data = {
                    "i" + str(i): np.arange(i, dtype=int),
                    "f" + str(i): np.arange(i, dtype=float),
                }
                kas.dump(data, f, engine=self.engine)

    def test_load_fileobj_single(self):
        data = {"a": np.arange(10)}
        kas.dump(data, self.temp_file, engine=self.engine)
        file_size = os.stat(self.temp_file).st_size
        for read_all in [True, False]:
            with open(self.temp_file, "rb") as f:
                data_out = kas.load(f, read_all=read_all, engine=self.engine)
                data2 = dict(data_out.items())
                file_offset = f.tell()
            self.verify_dicts_equal(data, data2)
            self.assertEqual(file_offset, file_size)

    def test_load_and_dump_fileobj_single(self):
        data = {"a": np.arange(10)}
        with open(self.temp_file, "wb") as f:
            kas.dump(data, f, engine=self.engine)
        file_size = os.stat(self.temp_file).st_size
        for read_all in [True, False]:
            with open(self.temp_file, "rb") as f:
                data_out = kas.load(f, read_all=read_all, engine=self.engine)
                data2 = dict(data_out.items())
                file_offset = f.tell()
            self.verify_dicts_equal(data, data2)
            self.assertEqual(file_offset, file_size)

    def test_load_and_dump_fileobj_multi(self):
        datalist = [
            {
                "i" + str(i): i + np.arange(10 ** 5, dtype=int),
                "f" + str(i): i + np.arange(10 ** 5, dtype=float),
            }
            for i in range(10)
        ]
        file_offsets = []
        with open(self.temp_file, "wb") as f:
            for data in datalist:
                kas.dump(data, f, engine=self.engine)
                file_offsets.append(f.tell())
        for read_all in [True, False]:
            with open(self.temp_file, "rb") as f:
                for data, file_offset in zip(datalist, file_offsets):
                    data_out = kas.load(f, read_all=read_all, engine=self.engine)
                    data2 = dict(data_out.items())
                    file_offset2 = f.tell()
                    self.verify_dicts_equal(data, data2)
                    self.assertEqual(file_offset, file_offset2)

    def test_load_and_dump_file_single_rw(self):
        data = {"a": np.arange(10)}
        with open(self.temp_file, "r+b") as f:
            kas.dump(data, f, engine=self.engine)
            for read_all in [True, False]:
                f.seek(0)
                data_out = kas.load(f, read_all=read_all, engine=self.engine)
                data2 = dict(data_out.items())
                self.verify_dicts_equal(data, data2)

    def test_load_and_dump_file_multi_rw(self):
        datalist = [
            {
                "i" + str(i): i + np.arange(10 ** 5, dtype=int),
                "f" + str(i): i + np.arange(10 ** 5, dtype=float),
            }
            for i in range(10)
        ]
        with open(self.temp_file, "r+b") as f:
            for data in datalist:
                kas.dump(data, f, engine=self.engine)

            for read_all in [True, False]:
                f.seek(0)
                for data in datalist:
                    data_out = kas.load(f, read_all=read_all, engine=self.engine)
                    data2 = dict(data_out.items())
                    self.verify_dicts_equal(data, data2)

    def test_load_and_dump_fd_single_rw(self):
        data = {"a": np.arange(10)}
        with open(self.temp_file, "r+b") as f:
            fd = f.fileno()
            kas.dump(data, fd, engine=self.engine)
            for read_all in [True, False]:
                os.lseek(fd, 0, os.SEEK_SET)
                data_out = kas.load(fd, read_all=read_all, engine=self.engine)
                data2 = dict(data_out.items())
                self.verify_dicts_equal(data, data2)

    def test_load_and_dump_fd_multi_rw(self):
        datalist = [
            {
                "i" + str(i): i + np.arange(10 ** 5, dtype=int),
                "f" + str(i): i + np.arange(10 ** 5, dtype=float),
            }
            for i in range(20)
        ]
        with open(self.temp_file, "r+b") as f:
            fd = f.fileno()
            for data in datalist:
                kas.dump(data, fd, engine=self.engine)
            for read_all in [True, False]:
                os.lseek(fd, 0, os.SEEK_SET)
                for data in datalist:
                    data_out = kas.load(fd, read_all=read_all, engine=self.engine)
                    data2 = dict(data_out.items())
                    self.verify_dicts_equal(data, data2)


def dump_to_stream(q_err, q_in, file_out, engine):
    """
    Get data dicts from `q_in` and kas.dump() them to `file_out`.
    Uncaught exceptions are placed onto the `q_err` queue.
    """
    try:
        with open(file_out, "wb") as f:
            while True:
                data = q_in.get()
                if data is None:
                    break
                kas.dump(data, f, engine=engine)
    except Exception as exc:
        tb = traceback.format_exc()
        q_err.put((exc, tb))


def load_from_stream(q_err, q_out, file_in, engine, read_all):
    """
    kas.load() stores from `file_in` and put their data onto `q_out`.
    Uncaught exceptions are placed onto the `q_err` queue.
    """
    try:
        with open(file_in, "rb") as f:
            while True:
                try:
                    data = kas.load(f, read_all=read_all, engine=engine)
                except EOFError:
                    break
                q_out.put(dict(data.items()))
    except Exception as exc:
        tb = traceback.format_exc()
        q_err.put((exc, tb))


class StreamingMixin(DictVerifyMixin):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="kas_test_streaming")
        self.temp_fifo = os.path.join(self.temp_dir, "fifo")
        os.mkfifo(self.temp_fifo)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def stream(self, datalist, read_all=True):
        """
        data -> q_in -> kas.dump(..., fifo) -> kas.load(fifo) -> q_out -> data_out
        """
        q_err = multiprocessing.Queue()
        q_in = multiprocessing.Queue()
        q_out = multiprocessing.Queue()
        proc1 = multiprocessing.Process(
            target=dump_to_stream, args=(q_err, q_in, self.temp_fifo, self.engine)
        )
        proc2 = multiprocessing.Process(
            target=load_from_stream,
            args=(q_err, q_out, self.temp_fifo, self.engine, read_all),
        )
        proc1.start()
        proc2.start()
        for data in datalist:
            q_in.put(data)
        q_in.put(None)  # signal the process that we're done
        proc1.join(timeout=3)
        if not q_err.empty():
            # re-raise the first child exception
            exc, tb = q_err.get()
            print(tb)
            raise exc
        if proc1.is_alive():
            # prevent hang if proc1 failed to join
            proc1.terminate()
            proc2.terminate()
            self.assertTrue(False, msg="proc1 (kas.dump) failed to join")
        datalist_out = []
        for _ in datalist:
            try:
                data_out = q_out.get(timeout=3)
            except queue.Empty:
                # terminate proc2 so we don't hang
                proc2.terminate()
                raise
            datalist_out.append(data_out)
        proc2.join(timeout=3)
        if proc2.is_alive():
            # prevent hang if proc2 failed to join
            proc2.terminate()
            self.assertTrue(False, msg="proc2 (kas.load) failed to join")

        self.assertEqual(len(datalist), len(datalist_out))
        for data, data_out in zip(datalist, datalist_out):
            self.verify_dicts_equal(data, data_out)

    def test_stream_single(self):
        datalist = [{"a": np.array([0])}]
        self.stream(datalist)

    def test_stream_multi(self):
        datalist = [
            {
                "i" + str(i): i + np.arange(10 ** 5, dtype=int),
                "f" + str(i): i + np.arange(10 ** 5, dtype=float),
            }
            for i in range(100)
        ]
        self.stream(datalist)


ADDRESS = ("localhost", 10009)


class TestServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True


class StoreEchoHandler(socketserver.BaseRequestHandler):
    def handle(self):
        while True:
            try:
                data = kas.load(
                    self.request.fileno(), engine=self.engine, read_all=True
                )
            except EOFError:
                break
            kas.dump(dict(data), self.request.fileno(), engine=self.engine)
        # We only read one list, so shutdown the server straight away
        self.server.shutdown()


class StoreEchoHandlerCEngine(StoreEchoHandler):
    engine = kas.C_ENGINE


class StoreEchoHandlerPyEngine(StoreEchoHandler):
    engine = kas.PY_ENGINE


def server_process(engine, q):
    handlers = {
        kas.C_ENGINE: StoreEchoHandlerCEngine,
        kas.PY_ENGINE: StoreEchoHandlerPyEngine,
    }
    server = TestServer(ADDRESS, handlers[engine])
    # Tell the client (on the other end of the queue) that it's OK to open
    # a connection
    q.put(None)
    server.serve_forever()


class SocketMixin(DictVerifyMixin):
    def setUp(self):
        # Use a queue to synchronise the startup of the server and the client.
        q = multiprocessing.Queue()
        self.server_process = multiprocessing.Process(
            target=server_process, args=(self.engine, q)
        )
        self.server_process.start()
        q.get()
        self.client = socket.create_connection(ADDRESS)

    def tearDown(self):
        self.client.close()
        self.server_process.join()

    def verify_stream(self, data_list):
        fd = self.client.fileno()
        for data in data_list:
            kas.dump(data, fd, engine=self.engine)
            echo_data = kas.load(fd, read_all=True, engine=self.engine)
            self.verify_dicts_equal(data, echo_data)

    def test_single(self):
        self.verify_stream([{"a": np.arange(10)}])

    def test_two(self):
        self.verify_stream([{"a": np.zeros(10)}, {"b": np.zeros(100)}])

    def test_multi(self):
        datalist = [
            {
                "i" + str(i): i + np.arange(10 ** 5, dtype=int),
                "f" + str(i): i + np.arange(10 ** 5, dtype=float),
            }
            for i in range(10)
        ]
        self.verify_stream(datalist)


class TestPathlibCEngine(PathlibMixin, unittest.TestCase):
    engine = kas.C_ENGINE


class TestPathlibPyEngine(PathlibMixin, unittest.TestCase):
    engine = kas.PY_ENGINE


class TestFileobjCEngine(FileobjMixin, unittest.TestCase):
    engine = kas.C_ENGINE


class TestFileobjPyEngine(FileobjMixin, unittest.TestCase):
    engine = kas.PY_ENGINE


@unittest.skipIf(IS_WINDOWS, "FIFOs don't exist on Windows")
class TestStreamingCEngine(StreamingMixin, unittest.TestCase):
    engine = kas.C_ENGINE


@unittest.skipIf(IS_WINDOWS, "FIFOs don't exist on Windows")
class TestStreamingPyEngine(StreamingMixin, unittest.TestCase):
    engine = kas.PY_ENGINE


@unittest.skipIf(IS_WINDOWS, "Deadlocking on Windows")
class TestSocketCEngine(SocketMixin, unittest.TestCase):
    engine = kas.C_ENGINE


@unittest.skipIf(IS_WINDOWS, "Deadlocking on Windows")
class TestSocketPyEngine(SocketMixin, unittest.TestCase):
    engine = kas.PY_ENGINE
