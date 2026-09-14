"""Reply keyboard labels stay aligned with the command handlers they dispatch to."""

from telegram_gateway.telegram.keyboard import BUTTON_COMMANDS, HIDE, main_keyboard


def test_every_button_except_hide_maps_to_a_command() -> None:
    labels = [button.text for row in main_keyboard().keyboard for button in row]
    assert labels[-1] == HIDE
    assert set(labels[:-1]) == set(BUTTON_COMMANDS)
