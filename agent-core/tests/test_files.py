"""Creating a file and sending it to Telegram is an MCP capability, not ACP Write."""

from __future__ import annotations

import pytest

from agent_core.assistant import prompts
from agent_core.agent.base import Provenance
from agent_core.calendar.local import LocalCalendarProvider
from agent_core.files import FileDelivery, looks_like_file_request, sanitize_filename
from agent_core.mcp.permissions import ToolContext
from agent_core.mcp.server import ContextRegistry, McpServer, ToolRegistry
from agent_core.mcp.tools import register_tools

from .test_mcp_server import TOKEN, Client, RecordingConfirmations


@pytest.mark.parametrize(
    ("raw", "want"),
    [
        ("план.csv", "план.csv"),
        ("notes/бюджет.md", "бюджет.md"),
        ("../etc/passwd", "passwd.md"),
        ("report", "report.md"),
        ("  weekly plan.CSV  ", "weekly plan.csv"),
        (r"C:\tmp\a.txt", "a.txt"),
    ],
)
def test_filename_is_reduced_to_a_safe_basename(raw: str, want: str) -> None:
    assert sanitize_filename(raw) == want


def test_binary_extensions_are_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        sanitize_filename("report.xlsx")
    with pytest.raises(ValueError, match="unsupported"):
        sanitize_filename("scan.pdf")


def test_empty_or_dot_names_are_rejected() -> None:
    with pytest.raises(ValueError):
        sanitize_filename("..")
    with pytest.raises(ValueError):
        sanitize_filename("///")


def test_the_session_preamble_tells_the_agent_to_use_file_send() -> None:
    text = prompts.session_preamble()
    assert "`file_send`" in text
    assert "PDF" in text
    assert "training_log_save" in text
    assert "журнал тренировок" in text
    assert "trainer" in text
    assert "training_progress" in text
    assert "осталось" in text


def test_file_intent_is_detected() -> None:
    assert looks_like_file_request("сделай отчет в маркдаун о планах")
    assert looks_like_file_request("пришли файл csv с бюджетом")
    assert looks_like_file_request("сохрани это в файл")
    assert not looks_like_file_request("а напоминания?")
    assert not looks_like_file_request("что у меня завтра")
    assert not looks_like_file_request("профайл на собесе")


@pytest.fixture
def delivery(settings, gateway) -> FileDelivery:
    return FileDelivery(gateway, settings.user_workspace)


def _ctx(**overrides) -> ToolContext:
    fields = dict(
        user_id="tg:1",
        conversation_id="conv",
        provenance=Provenance.DIRECT_COMMAND,
        job_id="job-1",
        chat_id=500,
        message_id=7,
    )
    fields.update(overrides)
    return ToolContext(**fields)


async def test_file_send_writes_to_the_sandbox_and_reaches_telegram(
    delivery, settings, gateway
) -> None:
    result = await delivery.send(
        _ctx(),
        filename="бюджет.csv",
        content="статья,сумма\nеда,12000\n",
        caption="Бюджет на август",
        operation_id="op-file-1",
    )

    assert result["sent"] is True
    assert result["filename"] == "бюджет.csv"
    saved = settings.user_workspace("tg:1") / "бюджет.csv"
    assert saved.read_text(encoding="utf-8") == "статья,сумма\nеда,12000\n"

    docs = gateway.documents()
    assert len(docs) == 1
    assert docs[0]["filename"] == "бюджет.csv"
    assert docs[0]["caption"] == "Бюджет на август"
    assert docs[0]["reply_to_message_id"] == 7
    assert "еда,12000" in docs[0]["content"]
    assert docs[0]["delivery_id"] == "job-1:file:op-file-1"
    assert (
        "telegram.action",
        {"chat_id": 500, "action": "upload_document"},
    ) in gateway.notifications


async def test_omitting_content_resends_an_existing_file(delivery, settings, gateway) -> None:
    await delivery.send(
        _ctx(), filename="заметка.md", content="# Черновик", caption=None, operation_id="op-a"
    )
    await delivery.send(
        _ctx(), filename="заметка.md", content=None, caption="Ещё раз", operation_id="op-b"
    )

    assert [d["filename"] for d in gateway.documents()] == ["заметка.md", "заметка.md"]
    assert gateway.documents()[1]["caption"] == "Ещё раз"
    assert "# Черновик" in gateway.documents()[1]["content"]


async def test_resending_a_missing_file_fails(delivery) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        await delivery.send(
            _ctx(), filename="нет.md", content=None, caption=None, operation_id="op-x"
        )


async def test_empty_content_is_rejected(delivery, gateway) -> None:
    with pytest.raises(ValueError, match="empty"):
        await delivery.send(
            _ctx(), filename="a.md", content="   \n", caption=None, operation_id="op-e"
        )
    assert gateway.documents() == []


async def test_nul_bytes_are_rejected(delivery) -> None:
    with pytest.raises(ValueError, match="NUL"):
        await delivery.send(
            _ctx(), filename="a.md", content="ok\x00no", caption=None, operation_id="op-n"
        )


def test_list_and_read_round_trip(delivery, settings) -> None:
    root = settings.user_workspace("tg:1")
    (root / "alpha.md").write_text("один\n", encoding="utf-8")
    (root / "beta.csv").write_text("a,b\n", encoding="utf-8")
    (root / ".hidden").write_text("nope\n", encoding="utf-8")

    listed = delivery.list_files("tg:1")
    assert [item["filename"] for item in listed["files"]] == ["alpha.md", "beta.csv"]

    read = delivery.read_file("tg:1", "alpha.md")
    assert read["content"] == "один\n"
    assert read["truncated"] is False
    assert delivery.read_file("tg:1", "../alpha.md")["content"] == "один\n"
    assert delivery.read_file("tg:1", "missing.md") == {"error": "not found"}


@pytest.fixture
async def files_mcp(repos, settings, gateway):
    registry = ToolRegistry()
    contexts = ContextRegistry()
    confirmations = RecordingConfirmations()
    register_tools(
        registry,
        repos,
        calendar_provider=LocalCalendarProvider(repos.calendar),
        file_delivery=FileDelivery(gateway, settings.user_workspace),
    )
    server = McpServer(
        registry, contexts, repos.operations, confirmations,
        host="127.0.0.1", port=0, token=TOKEN,
    )
    await server.start()
    try:
        yield server, contexts, confirmations, gateway, settings
    finally:
        await server.stop()


async def test_mcp_file_send_from_a_direct_command(files_mcp) -> None:
    server, contexts, confirmations, gateway, settings = files_mcp
    token = contexts.issue_token("conv")
    contexts.set_current("conv", _ctx())

    payload, is_error = await Client(server, token).call_tool(
        "file_send",
        {
            "filename": "список.md",
            "content": "- молоко\n- хлеб\n",
            "caption": "Покупки",
            "operation_id": "op-mcp-1",
        },
    )

    assert not is_error
    assert payload["filename"] == "список.md"
    assert confirmations.requests == []
    assert (settings.user_workspace("tg:1") / "список.md").is_file()
    assert gateway.documents()[0]["caption"] == "Покупки"


async def test_file_send_from_a_transcript_asks_first(files_mcp) -> None:
    server, contexts, confirmations, gateway, _settings = files_mcp
    token = contexts.issue_token("conv")
    contexts.set_current("conv", _ctx(provenance=Provenance.UNTRUSTED_CONTENT))

    payload, _ = await Client(server, token).call_tool(
        "file_send",
        {"filename": "из-записи.md", "content": "секрет", "operation_id": "op-mcp-2"},
    )

    assert payload["sent"] is True
    assert len(confirmations.requests) == 1
    assert confirmations.requests[0]["tool_name"] == "file_send"
    assert "из-записи.md" in confirmations.requests[0]["prompt_text"]
    assert len(gateway.documents()) == 1


async def test_replaying_file_send_does_not_deliver_twice(files_mcp) -> None:
    server, contexts, _confirmations, gateway, _settings = files_mcp
    token = contexts.issue_token("conv")
    contexts.set_current("conv", _ctx())
    client = Client(server, token)
    args = {
        "filename": "once.md",
        "content": "один раз",
        "operation_id": "op-mcp-once",
    }

    first, _ = await client.call_tool("file_send", args)
    second, _ = await client.call_tool("file_send", args)

    assert first == second
    assert len(gateway.documents()) == 1
