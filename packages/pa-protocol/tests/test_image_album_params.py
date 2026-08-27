"""ImageBegin album fields round-trip through dump/load."""

from __future__ import annotations

from pa_protocol import methods


def test_image_begin_album_fields_round_trip() -> None:
    params = methods.ImageBeginParams(
        request_id="req-1",
        user_id="tg:1",
        chat_id=1,
        message_id=2,
        filename="a.jpg",
        content_type="image/jpeg",
        size=12,
        purpose="ocr",
        album_id="album-1",
        part_index=1,
        part_count=3,
    )
    dumped = methods.dump(params)
    assert dumped["album_id"] == "album-1"
    assert dumped["part_index"] == 1
    assert dumped["part_count"] == 3

    loaded = methods.ImageBeginParams.model_validate(dumped)
    assert loaded.album_id == "album-1"
    assert loaded.part_index == 1
    assert loaded.part_count == 3


def test_image_begin_without_album_omits_fields() -> None:
    params = methods.ImageBeginParams(
        request_id="req-1",
        user_id="tg:1",
        chat_id=1,
        message_id=2,
        filename="a.jpg",
        content_type="image/jpeg",
        size=12,
    )
    dumped = methods.dump(params)
    assert "album_id" not in dumped
    assert "part_index" not in dumped
    assert "part_count" not in dumped
