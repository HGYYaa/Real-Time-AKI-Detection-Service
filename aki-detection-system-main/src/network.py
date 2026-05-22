from __future__ import annotations

import os
import time
import socket
import threading
import logging
from datetime import datetime, timezone
from typing import Callable, Optional, List

import requests

logger = logging.getLogger(__name__)

# --- MLLP Protocol Constants ---
# MLLP (Minimum Lower Layer Protocol) frames HL7 messages with specific bytes:
# <VT> (0x0b) + Message + <FS> (0x1c) + <CR> (0x0d)
_MLLP_START_BYTE = b"\x0b"
_MLLP_END_BYTE = b"\x1c"
_MLLP_CARRIAGE_RETURN = b"\x0d"

# --- Configuration Globals ---
_mllp_server_host: str = "127.0.0.1"
_mllp_server_port: int = 8440
_pager_base_url: str = "http://127.0.0.1:8441"
_pager_page_path: str = "/page"

# --- Network Tuning ---
_socket_receive_timeout_seconds: float = 10.0
_socket_receive_buffer_size_bytes: int = 65536

# --- Threading & State Control ---
_network_listener_thread: Optional[threading.Thread] = None
_stop_listening_flag: bool = False
_active_client_socket: Optional[socket.socket] = None
_listener_stopped_event = threading.Event()

# Callback function to hand off raw HL7 strings to the business logic
_pass_hl7_to_main_callback: Optional[Callable[[str], None]] = None


def initialize(pass_HL7: Callable[[str], None]) -> None:
    """
    Initializes the network layer and starts the background MLLP listener thread.

    Args:
        pass_HL7: Callback function that receives the raw HL7 message string.
                  This function is called immediately after a message is successfully
                  decoded from the MLLP stream.
    """
    global _pass_hl7_to_main_callback, _network_listener_thread, _stop_listening_flag
    global _mllp_server_host, _mllp_server_port, _pager_base_url, _pager_page_path

    _pass_hl7_to_main_callback = pass_HL7
    _stop_listening_flag = False
    _listener_stopped_event.clear()

    # Load configuration from environment variables (Cloud-Native friendly)
    mllp_address_text = os.getenv("MLLP_ADDRESS", "localhost:8440")
    _mllp_server_host, _mllp_server_port = _split_host_and_port(mllp_address_text)

    pager_address_text = os.getenv("PAGER_ADDRESS", "localhost:8441")
    _pager_base_url = _normalize_http_base_url(pager_address_text)
    _pager_page_path = os.getenv("PAGER_PAGE_PATH", "/page")

    logger.info("network.initialize: mllp=%s:%s pager=%s%s",
                _mllp_server_host, _mllp_server_port, _pager_base_url, _pager_page_path)

    # Idempotency check: prevent starting multiple listener threads
    if _network_listener_thread is None or not _network_listener_thread.is_alive():
        _network_listener_thread = threading.Thread(
            target=_network_listen_loop,
            name="mllp-network-listener",
            daemon=True
        )
        _network_listener_thread.start()


def wait(timeout_seconds: Optional[float] = None) -> bool:
    """
    Blocks the calling thread until the network listener stops or timeout expires.
    Useful for keeping the main program alive while the background thread works.
    """
    if timeout_seconds is None:
        return _listener_stopped_event.wait()
    return _listener_stopped_event.wait(timeout_seconds)


def alert(patient_id: str, timestamp_hl7: str) -> None:
    """
    Sends an alert to the external pager service via HTTP POST.

    Args:
        patient_id: The MRN or ID of the patient.
        timestamp_hl7: The formatted timestamp string from the HL7 message.
    """
    pager_endpoint_url = f"{_pager_base_url}{_pager_page_path}"
    # Simple CSV format body as expected by the pager simulator
    request_body_text = f"{patient_id},{timestamp_hl7}"

    try:
        response = requests.post(
            pager_endpoint_url,
            data=request_body_text.encode("utf-8"),
            timeout=2.0
        )
        if 200 <= response.status_code < 300:
            logger.info("alert: sent page ok patient_id=%s ts=%s", patient_id, timestamp_hl7)
        else:
            logger.warning("alert: pager returned non-2xx status=%s body=%s",
                           response.status_code, response.text[:200])
    except requests.RequestException as request_error:
        logger.exception("alert: failed to send page: %s", request_error)


def stop() -> None:
    """
    Signals the network listener to stop and forcibly closes the active socket.
    This ensures the recv() call unblocks immediately.
    """
    global _stop_listening_flag, _active_client_socket
    _stop_listening_flag = True

    if _active_client_socket is not None:
        try:
            _active_client_socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            _active_client_socket.close()
        except OSError:
            pass
        _active_client_socket = None


def _network_listen_loop() -> None:
    """
    Main loop for the background thread.
    Handles reconnection logic with exponential backoff if the connection drops.
    """
    global _stop_listening_flag

    reconnect_backoff_seconds = 1.0
    maximum_backoff_seconds = 5.0

    try:
        while not _stop_listening_flag:
            try:
                # Attempt to run a session. Returns True if closed cleanly by peer.
                ended_normally = _run_single_connection_session()

                if ended_normally:
                    # In this specific implementation (Pager Simulator),
                    # a clean disconnect implies the simulation round is over.
                    _stop_listening_flag = True
                    break

                # If error occurred, reset backoff for next clean attempt
                reconnect_backoff_seconds = 1.0
            except Exception as unexpected_error:
                logger.exception("network loop: unexpected error: %s", unexpected_error)

            if _stop_listening_flag:
                break

            # Reconnection Backoff Logic
            logger.warning("network loop: disconnected; retrying in %.1fs", reconnect_backoff_seconds)
            time.sleep(reconnect_backoff_seconds)
            reconnect_backoff_seconds = min(
                maximum_backoff_seconds,
                reconnect_backoff_seconds * 1.5
            )
    finally:
        _listener_stopped_event.set()


def _run_single_connection_session() -> bool:
    """
    Manages a single TCP connection lifecycle: Connect -> Loop(Recv/Process/Ack) -> Close.

    Returns:
        True: Peer closed connection cleanly (EOF).
        False: Socket error occurred (triggering reconnection).
    """
    global _active_client_socket

    if _pass_hl7_to_main_callback is None:
        raise RuntimeError("network.initialize must be called before listening")

    logger.info("connecting to mllp server %s:%s", _mllp_server_host, _mllp_server_port)

    try:
        with socket.create_connection(
                (_mllp_server_host, _mllp_server_port),
                timeout=_socket_receive_timeout_seconds
        ) as client_socket:
            _active_client_socket = client_socket
            client_socket.settimeout(_socket_receive_timeout_seconds)
            logger.info("connected to mllp server")

            receive_byte_buffer = bytearray()

            while not _stop_listening_flag:
                try:
                    received_chunk_bytes = client_socket.recv(_socket_receive_buffer_size_bytes)
                except socket.timeout:
                    continue  # Keep waiting, check stop flag
                except (ConnectionResetError, BrokenPipeError, OSError) as socket_error:
                    logger.warning("socket error during recv: %s", socket_error)
                    return False

                if not received_chunk_bytes:
                    logger.info("peer closed connection; treating as end-of-round (no reconnect)")
                    return True

                receive_byte_buffer.extend(received_chunk_bytes)

                # Process buffer: Extract complete HL7 messages framed by MLLP bytes

                for raw_hl7_text in _extract_complete_hl7_messages(receive_byte_buffer):
                    try:
                        # 1. Handoff to business logic
                        _pass_hl7_to_main_callback(raw_hl7_text)
                    except Exception as callback_error:
                        logger.exception("receive_HL7 callback failed: %s", callback_error)
                    finally:
                        # 2. Always send ACK (Accept) to keep the stream moving
                        try:
                            ack_hl7_text = _build_ack_message(ack_code="AA")
                            client_socket.sendall(_wrap_mllp_frame(ack_hl7_text))
                        except (BrokenPipeError, ConnectionResetError, OSError) as ack_error:
                            logger.warning("failed to send ACK: %s", ack_error)
                            return False

            return True
    finally:
        _active_client_socket = None


def _extract_complete_hl7_messages(receive_byte_buffer: bytearray) -> List[str]:
    """
    Parses the byte buffer to find and extract complete MLLP-framed messages.
    Handles TCP fragmentation (split packets) and coalescing (sticky packets).

    MLLP Frame: <SB> Payload <EB><CR>
    """
    extracted_hl7_messages: List[str] = []

    while True:
        if not receive_byte_buffer:
            return extracted_hl7_messages

        # 1. Find Start Byte
        try:
            start_index = receive_byte_buffer.index(_MLLP_START_BYTE)
        except ValueError:
            # No start byte found, discard garbage data to save memory
            receive_byte_buffer.clear()
            return extracted_hl7_messages

        # Discard data before the start byte
        if start_index > 0:
            del receive_byte_buffer[:start_index]

        # 2. Find End Byte
        try:
            end_index = receive_byte_buffer.index(_MLLP_END_BYTE, 1)
        except ValueError:
            # Packet is incomplete (split), wait for more data
            return extracted_hl7_messages

        # 3. Validation: Ensure <EB> is followed by <CR>
        if end_index + 1 >= len(receive_byte_buffer):
            # We have EB but not enough bytes to check for CR yet
            return extracted_hl7_messages

        if receive_byte_buffer[end_index + 1: end_index + 2] != _MLLP_CARRIAGE_RETURN:
            # Invalid frame (no CR after EB), skip this start byte and retry
            del receive_byte_buffer[:1]
            continue

        # 4. Extract Payload
        payload_bytes = bytes(receive_byte_buffer[1:end_index])

        # Remove processed message from buffer (including framing bytes)
        del receive_byte_buffer[: end_index + 2]

        extracted_hl7_messages.append(payload_bytes.decode("utf-8", errors="replace"))


def _wrap_mllp_frame(hl7_message_text: str) -> bytes:
    """Wraps an HL7 string in MLLP framing bytes for network transmission."""
    return _MLLP_START_BYTE + hl7_message_text.encode("utf-8") + _MLLP_END_BYTE + _MLLP_CARRIAGE_RETURN


def _build_ack_message(ack_code: str = "AA") -> str:
    """Constructs a minimal HL7 ACK message."""
    ack_timestamp_text = _format_hl7_timestamp(datetime.now(timezone.utc))
    return "\r".join([f"MSH|^~\\&|||||{ack_timestamp_text}||ACK|||2.5", f"MSA|{ack_code}", ])


def _format_hl7_timestamp(dt: datetime) -> str:
    """Formats datetime to HL7 standard: YYYYMMDDHHMMSS"""
    dt_utc = dt.astimezone(timezone.utc)
    return dt_utc.strftime("%Y%m%d%H%M%S")


def _split_host_and_port(host_port_text: str) -> tuple[str, int]:
    """Parses 'host:port' string into a tuple."""
    host_text, port_text = host_port_text.strip().rsplit(":", 1)
    return host_text.strip(), int(port_text.strip())


def _normalize_http_base_url(address_text: str) -> str:
    """Ensures URL has protocol schema and no trailing slash."""
    address_text = address_text.strip()
    if address_text.startswith("http://") or address_text.startswith("https://"):
        return address_text.rstrip("/")
    return f"http://{address_text}".rstrip("/")