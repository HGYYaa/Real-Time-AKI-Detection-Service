import pytest
import requests
from unittest.mock import Mock
from src import network


def _frame(text: str) -> bytes:
    return network._MLLP_START_BYTE + text.encode("utf-8") + network._MLLP_END_BYTE + network._MLLP_CARRIAGE_RETURN


def test_extract_single_message() -> None:
    buf = bytearray()
    buf.extend(_frame("MSH|A\rPID|1\r"))
    out = network._extract_complete_hl7_messages(buf)
    assert out == ["MSH|A\rPID|1\r"]
    assert buf == bytearray()


def test_extract_two_messages_sticky_packets() -> None:
    buf = bytearray()
    buf.extend(_frame("MSH|1\r"))
    buf.extend(_frame("MSH|2\r"))
    out = network._extract_complete_hl7_messages(buf)
    assert out == ["MSH|1\r", "MSH|2\r"]
    assert buf == bytearray()


def test_extract_partial_message_then_complete() -> None:
    msg = _frame("MSH|X\r")
    buf = bytearray()
    buf.extend(msg[:5])
    out1 = network._extract_complete_hl7_messages(buf)
    assert out1 == []
    assert buf != bytearray()

    buf.extend(msg[5:])
    out2 = network._extract_complete_hl7_messages(buf)
    assert out2 == ["MSH|X\r"]
    assert buf == bytearray()


def test_extract_garbage_before_start_byte_is_dropped() -> None:
    buf = bytearray(b"garbagegarbage")
    buf.extend(_frame("MSH|OK\r"))
    out = network._extract_complete_hl7_messages(buf)
    assert out == ["MSH|OK\r"]
    assert buf == bytearray()


def test_extract_invalid_end_sequence_skips_one_byte() -> None:
    payload = "MSH|BAD\r".encode("utf-8")
    buf = bytearray()
    buf.extend(network._MLLP_START_BYTE + payload + network._MLLP_END_BYTE + b"\x00")
    buf.extend(_frame("MSH|GOOD\r"))
    out = network._extract_complete_hl7_messages(buf)
    assert out == ["MSH|GOOD\r"]
    assert buf == bytearray()


def test_wrap_mllp_frame_roundtrip() -> None:
    text = "MSH|R\rPID|2\r"
    framed = network._wrap_mllp_frame(text)
    buf = bytearray(framed)
    out = network._extract_complete_hl7_messages(buf)
    assert out == [text]
    assert buf == bytearray()


def test_split_host_and_port() -> None:
    host, port = network._split_host_and_port("localhost:8440")
    assert host == "localhost"
    assert port == 8440


def test_normalize_http_base_url_with_scheme() -> None:
    assert network._normalize_http_base_url("http://localhost:8441") == "http://localhost:8441"
    assert network._normalize_http_base_url("https://cw3.com/") == "https://cw3.com"


def test_normalize_http_base_url_without_scheme() -> None:
    assert network._normalize_http_base_url("localhost:8441") == "http://localhost:8441"
    assert network._normalize_http_base_url("127.0.0.1:8441/") == "http://127.0.0.1:8441"


def test_build_ack_message_contains_msh_and_msa() -> None:
    ack = network._build_ack_message("AA")
    parts = ack.split("\r")
    assert len(parts) == 2
    assert parts[0].startswith("MSH|")
    assert parts[1] == "MSA|AA"


def test_alert_success(monkeypatch) -> None:
    network._pager_base_url = "http://pager"
    network._pager_page_path = "/page"

    fake_response = Mock()
    fake_response.status_code = 200
    fake_response.text = "ok"

    fake_post = Mock(return_value=fake_response)
    monkeypatch.setattr(network.requests, "post", fake_post)

    network.alert("123", "20260203120000")

    fake_post.assert_called_once()
    args, kwargs = fake_post.call_args
    assert args[0] == "http://pager/page"
    assert kwargs["timeout"] == 2.0
    assert kwargs["data"] == b"123,20260203120000"


def test_alert_non_2xx(monkeypatch) -> None:
    network._pager_base_url = "http://pager"
    network._pager_page_path = "/page"

    fake_response = Mock()
    fake_response.status_code = 500
    fake_response.text = "error"

    fake_post = Mock(return_value=fake_response)
    monkeypatch.setattr(network.requests, "post", fake_post)

    network.alert("p", "t")

    fake_post.assert_called_once()


def test_alert_request_exception(monkeypatch) -> None:
    network._pager_base_url = "http://pager"
    network._pager_page_path = "/page"

    def raise_error(*args, **kwargs):
        raise requests.RequestException("fail")

    monkeypatch.setattr(network.requests, "post", raise_error)

    network.alert("p", "t")


class _FakeSocket:
    def __init__(self, recv_chunks):
        self._recv_chunks = list(recv_chunks)
        self.sent = []
        self.timeout = None

    def settimeout(self, t):
        self.timeout = t

    def recv(self, n):
        if not self._recv_chunks:
            return b""
        return self._recv_chunks.pop(0)

    def sendall(self, data):
        self.sent.append(data)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_run_single_connection_session_calls_callback_and_sends_ack(monkeypatch) -> None:
    raw_hl7_text = "MSH|A\rPID|1\r"
    framed = network._wrap_mllp_frame(raw_hl7_text)

    fake_socket = _FakeSocket([framed, b""])
    monkeypatch.setattr(network.socket, "create_connection", lambda *args, **kwargs: fake_socket)

    received_messages = []

    def pass_HL7(msg: str) -> None:
        received_messages.append(msg)

    network._pass_hl7_to_main_callback = pass_HL7
    network._stop_listening_flag = False

    network._run_single_connection_session()

    assert received_messages == [raw_hl7_text]
    assert len(fake_socket.sent) == 1

    ack_buffer = bytearray(fake_socket.sent[0])
    ack_texts = network._extract_complete_hl7_messages(ack_buffer)
    assert len(ack_texts) == 1
    assert "MSH|" in ack_texts[0]
    assert "MSA|AA" in ack_texts[0]


class _FakeShutdownSocket:
    def __init__(self):
        self.shutdown_called = False
        self.close_called = False
        self.shutdown_arg = None

    def shutdown(self, how):
        self.shutdown_called = True
        self.shutdown_arg = how

    def close(self):
        self.close_called = True


def test_stop_shuts_down_and_closes_active_socket() -> None:
    fake_socket = _FakeShutdownSocket()
    network._active_client_socket = fake_socket
    network._stop_listening_flag = False

    network.stop()

    assert network._stop_listening_flag is True
    assert fake_socket.shutdown_called is True
    assert fake_socket.shutdown_arg == network.socket.SHUT_RDWR
    assert fake_socket.close_called is True
    assert network._active_client_socket is None


class _FakeBrokenSocket:
    def shutdown(self, how):
        raise OSError("boom")

    def close(self):
        raise OSError("boom")


def test_stop_ignores_socket_errors() -> None:
    network._active_client_socket = _FakeBrokenSocket()
    network._stop_listening_flag = False

    network.stop()

    assert network._stop_listening_flag is True
    assert network._active_client_socket is None