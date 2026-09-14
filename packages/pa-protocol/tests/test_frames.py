from __future__ import annotations

import pytest

from pa_protocol import new_ulid
from pa_protocol.frames import (
    HEADER_SIZE,
    AudioFrame,
    FrameError,
    decode_frame,
    encode_frame,
    iter_frames,
)
from pa_protocol.ids import ulid_from_bytes, ulid_to_bytes


def test_ulid_roundtrips_through_binary():
    value = new_ulid()
    assert ulid_from_bytes(ulid_to_bytes(value)) == value


def test_ulids_sort_by_creation_order():
    values = [new_ulid() for _ in range(500)]
    assert values == sorted(values), "ULIDs must be monotonic to allow ordered event replay"
    assert len(set(values)) == len(values)


def test_frame_roundtrip_preserves_payload_offset_and_final_flag():
    upload_id = new_ulid()
    frame = AudioFrame(upload_id=upload_id, offset=1024, payload=b"\x00\xffbytes", final=True)
    decoded = decode_frame(encode_frame(frame))
    assert decoded == frame
    assert decoded.end_offset == 1024 + len(frame.payload)


def test_frame_header_is_exactly_32_bytes():
    encoded = encode_frame(AudioFrame(new_ulid(), 0, b""))
    assert len(encoded) == HEADER_SIZE


@pytest.mark.parametrize(
    "corrupt",
    [
        b"",
        b"short",
        b"XXXX" + b"\x00" * 28,
    ],
)
def test_undecodable_frames_raise_rather_than_returning_garbage(corrupt):
    with pytest.raises(FrameError):
        decode_frame(corrupt)


def test_iter_frames_covers_all_bytes_in_order_with_one_final_frame():
    upload_id = new_ulid()
    data = bytes(range(256)) * 40  # 10240 bytes
    frames = [decode_frame(raw) for raw in iter_frames(upload_id, data, chunk_size=1000)]

    assert [f.offset for f in frames] == list(range(0, 10240, 1000))
    assert sum(f.final for f in frames) == 1
    assert frames[-1].final

    reassembled = bytearray(len(data))
    for frame in frames:
        reassembled[frame.offset : frame.end_offset] = frame.payload
    assert bytes(reassembled) == data


def test_empty_upload_still_produces_a_terminating_frame():
    frames = list(iter_frames(new_ulid(), b""))
    assert len(frames) == 1
    assert decode_frame(frames[0]).final


def test_start_offset_supports_resuming_an_interrupted_upload():
    upload_id = new_ulid()
    frames = [decode_frame(f) for f in iter_frames(upload_id, b"tail", start_offset=4096)]
    assert frames[0].offset == 4096
