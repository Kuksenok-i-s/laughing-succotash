"""HTTP contract the Core depends on."""

from __future__ import annotations

import http.client
import json

from helpers import TOKEN, Client, FakeEngine, wait_for

from handwriting_ocr.jobs import JobStore
from handwriting_ocr.server import OcrApp
from handwriting_ocr.worker import OcrWorker


def test_health_needs_no_token_and_admits_a_cold_model(client: Client) -> None:
    response = client.request("GET", "/health", token=None)

    assert response.status == 200
    assert response.payload["status"] == "ok"
    assert response.payload["model"] == "fake-qwen3-vl"
    assert response.payload["model_loaded"] is False
    assert response.payload["queued"] == 0


def test_everything_else_requires_the_token(client: Client) -> None:
    assert client.request("GET", "/v1/jobs/abc", token=None).status == 401
    assert client.request("GET", "/v1/jobs/abc", token="wrong").status == 401
    assert client.request("PUT", "/v1/jobs/abc", body=b"x", token="wrong").status == 401
    assert client.request("DELETE", "/v1/jobs/abc", token="wrong").status == 401
    assert client.request("POST", "/v1/model/load", token="wrong").status == 401
    assert client.request("GET", "/v1/jobs/abc", token=TOKEN).status == 404


def test_an_upload_queues_a_job_and_spools_the_image(client: Client, store: JobStore) -> None:
    response = client.put_image("01IMG", b"jpeg-bytes", filename="note.jpg", content_type="image/jpeg")

    assert response.status == 202
    assert response.payload["status"] == "queued"
    job = store.get("01IMG")
    assert job is not None
    assert job.image_path.read_bytes() == b"jpeg-bytes"
    assert store.queue_depth() == 1


def test_a_repeated_upload_does_not_start_a_second_job(client: Client, store: JobStore) -> None:
    first = client.put_image("01SAME", b"jpeg-bytes")
    second = client.put_image("01SAME", b"jpeg-bytes")

    assert first.status == 202
    assert second.status == 200
    assert store.queue_depth() == 1


def test_a_non_image_content_type_is_refused(client: Client, store: JobStore) -> None:
    response = client.put_image("01PDF", b"%PDF", content_type="application/pdf")

    assert response.status == 415
    assert response.payload["code"] == "invalid_image"
    assert store.get("01PDF") is None


def test_an_oversized_upload_is_refused_without_reading_the_body(
    client: Client, store: JobStore
) -> None:
    connection = http.client.HTTPConnection(client.host, client.port, timeout=10)
    try:
        connection.putrequest("PUT", "/v1/jobs/01BIG", skip_accept_encoding=True)
        connection.putheader("Authorization", f"Bearer {TOKEN}")
        connection.putheader("Content-Type", "image/jpeg")
        connection.putheader("Content-Length", str(600 * 1024 * 1024))
        connection.putheader("Expect", "100-continue")
        connection.endheaders()
        response = connection.getresponse()
        status = response.status
        payload = json.loads(response.read())
    finally:
        connection.close()

    assert status == 413
    assert payload["code"] == "image_too_large"
    assert store.get("01BIG") is None


def test_two_pass_recognition_feeds_raw_text_into_pass_two(
    settings, store: JobStore, engine: FakeEngine, serve
) -> None:
    engine.load()
    app = OcrApp(settings, store, engine)
    client = serve(app)
    worker = OcrWorker(store, engine)

    client.put_image("01PASS", b"jpeg-bytes")
    worker.run_job("01PASS")

    assert wait_for(lambda: store.get("01PASS").status == "done")
    assert len(engine.calls) == 3
    assert engine.calls[0]["stage"] == "triage"
    assert engine.calls[1]["stage"] == "refine"
    assert engine.calls[1]["raw_text"] == engine._raw_text
    assert engine.calls[2]["stage"] == "structure"

    result = client.request("GET", "/v1/jobs/01PASS/result")
    assert result.status == 200
    assert result.payload["raw_text"] == engine._raw_text
    assert result.payload["markdown"] == engine._markdown
    assert result.payload["passes"] == 3


def test_model_load_and_unload(client: Client, engine: FakeEngine) -> None:
    loaded = client.request("POST", "/v1/model/load", body=b"")
    unloaded = client.request("POST", "/v1/model/unload", body=b"")

    assert loaded.status == 200
    assert loaded.payload["model_loaded"] is True
    assert unloaded.status == 200
    assert unloaded.payload["model_loaded"] is False
    assert engine.loads == 1
    assert engine.unloads == 1


def test_delete_forgets_the_job_and_removes_the_image(client: Client, store: JobStore) -> None:
    client.put_image("01GONE")
    spool = store.spool_dir("01GONE")
    assert spool.exists()

    first = client.request("DELETE", "/v1/jobs/01GONE")
    second = client.request("DELETE", "/v1/jobs/01GONE")

    assert first.status == 200
    assert second.status == 404
    assert not spool.exists()


def test_a_failed_job_reports_its_reason(client: Client, store: JobStore) -> None:
    client.put_image("01BAD")
    store.fail("01BAD", "RuntimeError: ollama is on fire")

    status = client.request("GET", "/v1/jobs/01BAD")
    result = client.request("GET", "/v1/jobs/01BAD/result")

    assert status.payload["status"] == "failed"
    assert result.status == 409
    assert result.payload["code"] == "job_failed"
