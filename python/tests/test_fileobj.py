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

import numpy as np
import pytest

import kastore as kas

"""
Tests for load()ing and dump()ing with file-like objects.
"""


IS_WINDOWS = platform.system() == "Windows"


def verify_dicts_equal(d1, d2):
    assert sorted(d1.keys()) == sorted(d2.keys())
    for key in d1.keys():
        np.testing.assert_equal(d1[key], d2[key])


# pathlib.Path objects should work transparently, and these tests check it
# isn't broken by fileobj-enabling code.
@pytest.fixture
def pathlib_temp_file():
    fd, path = tempfile.mkstemp(prefix="kas_test_pathlib")
    os.close(fd)
    temp_file = pathlib.Path(path)
    yield temp_file
    temp_file.unlink()


@pytest.mark.parametrize("engine", [kas.PY_ENGINE, kas.C_ENGINE])
def test_dump_to_pathlib_path(pathlib_temp_file, engine):
    data = {"a": np.arange(10)}
    kas.dump(data, pathlib_temp_file, engine=engine)


@pytest.mark.parametrize("engine", [kas.PY_ENGINE, kas.C_ENGINE])
@pytest.mark.parametrize("read_all", [True, False])
def test_load_from_pathlib_path(pathlib_temp_file, engine, read_all):
    data = {"a": np.arange(10)}
    kas.dump(data, str(pathlib_temp_file), engine=engine)
    file_size = pathlib_temp_file.stat().st_size
    data_out = kas.load(pathlib_temp_file, read_all=read_all, engine=engine)
    data2 = dict(data_out.items())
    file_size2 = pathlib_temp_file.stat().st_size
    verify_dicts_equal(data, data2)
    assert file_size == file_size2


@pytest.fixture
def fileobj_temp_file():
    fd, path = tempfile.mkstemp(prefix="kas_test_fileobj")
    os.close(fd)
    yield path
    os.unlink(path)


@pytest.mark.parametrize("engine", [kas.PY_ENGINE, kas.C_ENGINE])
def test_dump_fileobj_single(fileobj_temp_file, engine):
    data = {"a": np.arange(10)}
    with open(fileobj_temp_file, "wb") as f:
        kas.dump(data, f, engine=engine)
    data_out = kas.load(fileobj_temp_file, engine=engine)
    data2 = dict(data_out.items())
    verify_dicts_equal(data, data2)


@pytest.mark.parametrize("engine", [kas.PY_ENGINE, kas.C_ENGINE])
def test_dump_fileobj_multi(fileobj_temp_file, engine):
    with open(fileobj_temp_file, "wb") as f:
        for i in range(10):
            data = {
                "i" + str(i): np.arange(i, dtype=int),
                "f" + str(i): np.arange(i, dtype=float),
            }
            kas.dump(data, f, engine=engine)


@pytest.mark.parametrize("engine", [kas.PY_ENGINE, kas.C_ENGINE])
@pytest.mark.parametrize("read_all", [True, False])
def test_load_fileobj_single(fileobj_temp_file, engine, read_all):
    data = {"a": np.arange(10)}
    kas.dump(data, fileobj_temp_file, engine=engine)
    file_size = os.stat(fileobj_temp_file).st_size
    with open(fileobj_temp_file, "rb") as f:
        data_out = kas.load(f, read_all=read_all, engine=engine)
        data2 = dict(data_out.items())
        file_offset = f.tell()
    verify_dicts_equal(data, data2)
    assert file_offset == file_size


@pytest.mark.parametrize("engine", [kas.PY_ENGINE, kas.C_ENGINE])
@pytest.mark.parametrize("read_all", [True, False])
def test_load_and_dump_fileobj_single(fileobj_temp_file, engine, read_all):
    data = {"a": np.arange(10)}
    with open(fileobj_temp_file, "wb") as f:
        kas.dump(data, f, engine=engine)
    file_size = os.stat(fileobj_temp_file).st_size
    with open(fileobj_temp_file, "rb") as f:
        data_out = kas.load(f, read_all=read_all, engine=engine)
        data2 = dict(data_out.items())
        file_offset = f.tell()
    verify_dicts_equal(data, data2)
    assert file_offset == file_size


@pytest.mark.parametrize("engine", [kas.PY_ENGINE, kas.C_ENGINE])
@pytest.mark.parametrize("read_all", [True, False])
def test_load_and_dump_fileobj_multi(fileobj_temp_file, engine, read_all):
    datalist = [
        {
            "i" + str(i): i + np.arange(10**5, dtype=int),
            "f" + str(i): i + np.arange(10**5, dtype=float),
        }
        for i in range(10)
    ]
    file_offsets = []
    with open(fileobj_temp_file, "wb") as f:
        for data in datalist:
            kas.dump(data, f, engine=engine)
            file_offsets.append(f.tell())
    with open(fileobj_temp_file, "rb") as f:
        for data, file_offset in zip(datalist, file_offsets):
            data_out = kas.load(f, read_all=read_all, engine=engine)
            data2 = dict(data_out.items())
            file_offset2 = f.tell()
            verify_dicts_equal(data, data2)
            assert file_offset == file_offset2


@pytest.mark.parametrize("engine", [kas.PY_ENGINE, kas.C_ENGINE])
@pytest.mark.parametrize("read_all", [True, False])
def test_load_and_dump_file_single_rw(fileobj_temp_file, engine, read_all):
    data = {"a": np.arange(10)}
    with open(fileobj_temp_file, "r+b") as f:
        kas.dump(data, f, engine=engine)
        f.seek(0)
        data_out = kas.load(f, read_all=read_all, engine=engine)
        data2 = dict(data_out.items())
        verify_dicts_equal(data, data2)


@pytest.mark.parametrize("engine", [kas.PY_ENGINE, kas.C_ENGINE])
@pytest.mark.parametrize("read_all", [True, False])
def test_load_and_dump_file_multi_rw(fileobj_temp_file, engine, read_all):
    datalist = [
        {
            "i" + str(i): i + np.arange(10**5, dtype=int),
            "f" + str(i): i + np.arange(10**5, dtype=float),
        }
        for i in range(10)
    ]
    with open(fileobj_temp_file, "r+b") as f:
        for data in datalist:
            kas.dump(data, f, engine=engine)

        f.seek(0)
        for data in datalist:
            data_out = kas.load(f, read_all=read_all, engine=engine)
            data2 = dict(data_out.items())
            verify_dicts_equal(data, data2)


@pytest.mark.parametrize("engine", [kas.PY_ENGINE, kas.C_ENGINE])
@pytest.mark.parametrize("read_all", [True, False])
def test_load_and_dump_fd_single_rw(fileobj_temp_file, engine, read_all):
    data = {"a": np.arange(10)}
    with open(fileobj_temp_file, "r+b") as f:
        fd = f.fileno()
        kas.dump(data, fd, engine=engine)
        os.lseek(fd, 0, os.SEEK_SET)
        data_out = kas.load(fd, read_all=read_all, engine=engine)
        data2 = dict(data_out.items())
        verify_dicts_equal(data, data2)


@pytest.mark.parametrize("engine", [kas.PY_ENGINE, kas.C_ENGINE])
@pytest.mark.parametrize("read_all", [True, False])
def test_load_and_dump_fd_multi_rw(fileobj_temp_file, engine, read_all):
    datalist = [
        {
            "i" + str(i): i + np.arange(10**5, dtype=int),
            "f" + str(i): i + np.arange(10**5, dtype=float),
        }
        for i in range(20)
    ]
    with open(fileobj_temp_file, "r+b") as f:
        fd = f.fileno()
        for data in datalist:
            kas.dump(data, fd, engine=engine)
        os.lseek(fd, 0, os.SEEK_SET)
        for data in datalist:
            data_out = kas.load(fd, read_all=read_all, engine=engine)
            data2 = dict(data_out.items())
            verify_dicts_equal(data, data2)


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


@pytest.fixture
def streaming_temp_fifo():
    temp_dir = tempfile.mkdtemp(prefix="kas_test_streaming")
    temp_fifo = os.path.join(temp_dir, "fifo")
    os.mkfifo(temp_fifo)
    yield temp_fifo
    shutil.rmtree(temp_dir)


def stream_data(temp_fifo, datalist, engine, read_all=True):
    """
    data -> q_in -> kas.dump(..., fifo) -> kas.load(fifo) -> q_out -> data_out
    """
    q_err = multiprocessing.Queue()
    q_in = multiprocessing.Queue()
    q_out = multiprocessing.Queue()
    proc1 = multiprocessing.Process(
        target=dump_to_stream, args=(q_err, q_in, temp_fifo, engine)
    )
    proc2 = multiprocessing.Process(
        target=load_from_stream,
        args=(q_err, q_out, temp_fifo, engine, read_all),
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
        raise AssertionError("proc1 (kas.dump) failed to join")
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
        raise AssertionError("proc2 (kas.load) failed to join")

    assert len(datalist) == len(datalist_out)
    for data, data_out in zip(datalist, datalist_out):
        verify_dicts_equal(data, data_out)


@pytest.mark.skipif(IS_WINDOWS, reason="FIFOs don't exist on Windows")
@pytest.mark.parametrize("engine", [kas.PY_ENGINE, kas.C_ENGINE])
def test_stream_single(streaming_temp_fifo, engine):
    datalist = [{"a": np.array([0])}]
    stream_data(streaming_temp_fifo, datalist, engine)


@pytest.mark.skipif(IS_WINDOWS, reason="FIFOs don't exist on Windows")
@pytest.mark.parametrize("engine", [kas.PY_ENGINE, kas.C_ENGINE])
def test_stream_multi(streaming_temp_fifo, engine):
    datalist = [
        {
            "i" + str(i): i + np.arange(10**5, dtype=int),
            "f" + str(i): i + np.arange(10**5, dtype=float),
        }
        for i in range(100)
    ]
    stream_data(streaming_temp_fifo, datalist, engine)


ADDRESS = ("localhost", 10009)


class HttpServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True


class StoreEchoHandler(socketserver.BaseRequestHandler):
    def handle(self):
        while True:
            try:
                data = kas.load(self.request.fileno(), engine=self.engine, read_all=True)
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
    server = HttpServer(ADDRESS, handlers[engine])
    # Tell the client (on the other end of the queue) that it's OK to open
    # a connection
    q.put(None)
    server.serve_forever()


@pytest.fixture
def socket_client(engine):
    # Use a queue to synchronise the startup of the server and the client.
    q = multiprocessing.Queue()
    server_proc = multiprocessing.Process(target=server_process, args=(engine, q))
    server_proc.start()
    q.get()
    client = socket.create_connection(ADDRESS)
    yield client, engine
    client.close()
    server_proc.join()


def verify_socket_stream(client, engine, data_list):
    fd = client.fileno()
    for data in data_list:
        kas.dump(data, fd, engine=engine)
        echo_data = kas.load(fd, read_all=True, engine=engine)
        verify_dicts_equal(data, echo_data)


@pytest.mark.skipif(IS_WINDOWS, reason="Deadlocking on Windows")
@pytest.mark.parametrize("engine", [kas.PY_ENGINE, kas.C_ENGINE])
def test_socket_single(socket_client):
    client, engine = socket_client
    verify_socket_stream(client, engine, [{"a": np.arange(10)}])


@pytest.mark.skipif(IS_WINDOWS, reason="Deadlocking on Windows")
@pytest.mark.parametrize("engine", [kas.PY_ENGINE, kas.C_ENGINE])
def test_socket_two(socket_client):
    client, engine = socket_client
    verify_socket_stream(client, engine, [{"a": np.zeros(10)}, {"b": np.zeros(100)}])


@pytest.mark.skipif(IS_WINDOWS, reason="Deadlocking on Windows")
@pytest.mark.parametrize("engine", [kas.PY_ENGINE, kas.C_ENGINE])
def test_socket_multi(socket_client):
    client, engine = socket_client
    datalist = [
        {
            "i" + str(i): i + np.arange(10**5, dtype=int),
            "f" + str(i): i + np.arange(10**5, dtype=float),
        }
        for i in range(10)
    ]
    verify_socket_stream(client, engine, datalist)
