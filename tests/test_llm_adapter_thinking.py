# -*- coding: utf-8 -*-
"""Tests for provider thinking-mode request payloads."""

from tests.litellm_stub import ensure_litellm_stub

ensure_litellm_stub()

from src.agent.llm_adapter import get_thinking_extra_body


def test_deepseek_v4_flash_enables_thinking_extra_body() -> None:
    assert get_thinking_extra_body("deepseek-v4-flash") == {"thinking": {"type": "enabled"}}
    assert get_thinking_extra_body("deepseek-v4-flash-202607") == {"thinking": {"type": "enabled"}}


def test_deepseek_v4_pro_keeps_current_non_opt_in_behavior() -> None:
    assert get_thinking_extra_body("deepseek-v4-pro") is None
