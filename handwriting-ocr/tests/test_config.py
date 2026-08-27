from handwriting_ocr.config import from_env


def test_from_env_reads_ocr_prefix() -> None:
    settings = from_env(
        {
            "OCR_TOKEN": "t" * 40,
            "OCR_HOST": "10.0.7.49",
            "OCR_PORT": "17494",
            "OCR_MODEL": "qwen3-vl:latest",
            "OCR_OLLAMA_URL": "http://127.0.0.1:11434",
            "OCR_OLLAMA_KEEP_ALIVE": "2m",
            "OCR_IDLE_UNLOAD_SECONDS": "90",
            "OCR_MAX_UPLOAD_MB": "16",
        }
    )

    assert settings.host == "10.0.7.49"
    assert settings.port == 17494
    assert settings.model == "qwen3-vl:latest"
    assert settings.keep_alive == "2m"
    assert settings.idle_unload_seconds == 90.0
    assert settings.max_upload_mb == 16
    assert settings.validate_runtime() == []


def test_defaults_idle_unload_is_ten_minutes() -> None:
    settings = from_env({"OCR_TOKEN": "t" * 40})
    assert settings.keep_alive == "10m"
    assert settings.idle_unload_seconds == 600.0


def test_missing_token_is_a_runtime_problem() -> None:
    settings = from_env({"OCR_PORT": "", "OCR_HOST": ""})
    assert any("TOKEN" in problem for problem in settings.validate_runtime())
