"""The HTTP contract the Core depends on."""

from __future__ import annotations

import http.client
import json
import logging

import pytest
from helpers import TOKEN, Client

from gpu_transcriber.jobs import JobStore
from gpu_transcriber.server import TranscriptionApp, TranscriptionServer


def test_health_needs_no_token_and_admits_a_cold_model(client: Client) -> None:
    response = client.request("GET", "/health", token=None)

    assert response.status == 200
    assert response.payload == {
        "status": "ok",
        "model": "fake-large-v3",
        "model_loaded": False,
        "queued": 0,
    }


def test_everything_else_requires_the_token(client: Client) -> None:
    assert client.request("GET", "/v1/jobs/abc", token=None).status == 401
    assert client.request("GET", "/v1/jobs/abc", token="wrong").status == 401
    assert client.request("PUT", "/v1/jobs/abc", body=b"x", token="wrong").status == 401
    assert client.request("DELETE", "/v1/jobs/abc", token="wrong").status == 401
    assert client.request("GET", "/v1/jobs/abc", token=TOKEN).status == 404


def test_an_upload_queues_a_job_and_spools_the_audio(client: Client, store: JobStore) -> None:
    response = client.put_audio("01ABC", b"audio bytes", language="ru", beam_size="3")

    assert response.status == 202
    assert response.payload["status"] == "queued"
    job = store.get("01ABC")
    assert job is not None
    assert job.language == "ru"
    assert job.beam_size == 3
    assert job.audio_path.read_bytes() == b"audio bytes"
    assert store.queue_depth() == 1


def test_auto_means_detect_the_language(client: Client, store: JobStore) -> None:
    """The SSH pipeline could only pass a fixed string, so every job claimed to be Russian."""
    client.put_audio("01AUTO", language="auto")

    assert store.get("01AUTO").language is None


def test_a_repeated_upload_does_not_start_a_second_job(client: Client, store: JobStore) -> None:
    first = client.put_audio("01SAME", b"audio bytes")
    second = client.put_audio("01SAME", b"audio bytes")

    assert first.status == 202
    assert second.status == 200
    assert second.payload["job_id"] == "01SAME"
    assert store.queue_depth() == 1


def test_a_job_id_that_is_not_a_plain_name_is_refused(client: Client) -> None:
    """The id becomes a directory name, so anything that could leave the work dir is rejected."""
    assert client.request("GET", "/v1/jobs/..").status == 400
    assert client.request("DELETE", "/v1/jobs/.").status == 400
    assert client.request("GET", "/v1/jobs/%2e%2e%2fetc").status == 400
    assert client.put_audio("01ABC.mp3").status == 400


def test_an_oversized_upload_is_refused_without_reading_the_body(
    client: Client, store: JobStore
) -> None:
    """With 100-continue the refusal arrives before a single byte of a half-gigabyte upload."""
    connection = http.client.HTTPConnection(client.host, client.port, timeout=10)
    try:
        connection.putrequest("PUT", "/v1/jobs/01BIG", skip_accept_encoding=True)
        connection.putheader("Authorization", f"Bearer {TOKEN}")
        connection.putheader("Content-Length", str(600 * 1024 * 1024))
        connection.putheader("Expect", "100-continue")
        connection.endheaders()
        response = connection.getresponse()
        status = response.status
        payload = json.loads(response.read())
    finally:
        connection.close()

    assert status == 413
    assert payload["code"] == "audio_too_large"
    assert store.get("01BIG") is None
    assert not store.spool_dir("01BIG").exists()


def test_an_upload_with_a_bad_token_is_refused_before_the_body_too(client: Client) -> None:
    connection = http.client.HTTPConnection(client.host, client.port, timeout=10)
    try:
        connection.putrequest("PUT", "/v1/jobs/01NOAUTH", skip_accept_encoding=True)
        connection.putheader("Authorization", "Bearer wrong")
        connection.putheader("Content-Length", "1024")
        connection.putheader("Expect", "100-continue")
        connection.endheaders()
        status = connection.getresponse().status
    finally:
        connection.close()

    assert status == 401


def test_an_oversized_upload_never_reaches_the_disk(client: Client, store: JobStore) -> None:
    """A client that ignores 100-continue still gets refused; it may see the closed connection
    rather than the 413, which is why the state of the service is what is asserted here."""
    try:
        response = client.put_audio("01BIG2", b"x" * (2 * 1024 * 1024))
        assert response.status == 413
    except (BrokenPipeError, ConnectionResetError):
        pass

    assert store.get("01BIG2") is None
    assert not store.spool_dir("01BIG2").exists()


def test_an_upload_without_a_length_is_refused(client: Client) -> None:
    connection = http.client.HTTPConnection(client.host, client.port, timeout=10)
    try:
        connection.putrequest("PUT", "/v1/jobs/01NOLEN", skip_accept_encoding=True)
        connection.putheader("Authorization", f"Bearer {TOKEN}")
        connection.endheaders()
        status = connection.getresponse().status
    finally:
        connection.close()

    assert status == 411


def test_the_result_says_what_the_job_is_doing_until_it_is_ready(
    client: Client, store: JobStore
) -> None:
    client.put_audio("01WAIT")
    store.report("01WAIT", percent=42.0, position_sec=25.0, duration_sec=60.0, segment_count=3)

    response = client.request("GET", "/v1/jobs/01WAIT/result")

    assert response.status == 409
    assert response.payload["percent"] == 42.0
    assert response.payload["segments"] == 3


def test_a_failed_job_reports_its_reason(client: Client, store: JobStore) -> None:
    client.put_audio("01BAD")
    store.fail("01BAD", "RuntimeError: cuda is on fire")

    status = client.request("GET", "/v1/jobs/01BAD")
    result = client.request("GET", "/v1/jobs/01BAD/result")

    assert status.payload["status"] == "failed"
    assert status.payload["error"] == "RuntimeError: cuda is on fire"
    assert result.status == 409
    assert result.payload["code"] == "job_failed"


def test_delete_forgets_the_job_and_removes_the_audio(client: Client, store: JobStore) -> None:
    client.put_audio("01GONE")
    spool = store.spool_dir("01GONE")
    assert spool.exists()

    first = client.request("DELETE", "/v1/jobs/01GONE")
    second = client.request("DELETE", "/v1/jobs/01GONE")

    assert first.status == 200
    assert second.status == 404
    assert not spool.exists()


def test_an_unknown_endpoint_is_a_404(client: Client) -> None:
    assert client.request("GET", "/v1/whatever").status == 404
    assert client.request("PUT", "/v1/whatever", body=b"x").status == 404


def test_a_client_that_hangs_up_is_not_logged_as_a_failure(
    app: TranscriptionApp, caplog: pytest.LogCaptureFixture
) -> None:
    """A transcript body is big enough that any interrupted read used to print a traceback."""
    server = TranscriptionServer(("127.0.0.1", 0), app)
    try:
        with caplog.at_level(logging.DEBUG, logger="gpu_transcriber.server"):
            try:
                raise ConnectionResetError(104, "Connection reset by peer")
            except ConnectionResetError:
                server.handle_error(None, ("127.0.0.1", 5000))
    finally:
        server.server_close()

    assert [record.levelno for record in caplog.records] == [logging.DEBUG]
