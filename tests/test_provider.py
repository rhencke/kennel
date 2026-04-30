from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from fido.provider import (
    ProviderID,
    ProviderInterruptTimeout,
    ProviderLimitSnapshot,
    ProviderLimitWindow,
    ProviderModel,
    ProviderPressureStatus,
    is_recoverable_provider_wedge,
    safe_voice_turn,
)


class TestProviderLimitWindow:
    def test_pressure_returns_ratio(self) -> None:
        window = ProviderLimitWindow(name="requests", used=9, limit=10)
        assert window.pressure == 0.9

    def test_pressure_returns_none_when_used_missing(self) -> None:
        window = ProviderLimitWindow(name="requests", used=None, limit=10)
        assert window.pressure is None

    def test_pressure_returns_none_when_limit_missing(self) -> None:
        window = ProviderLimitWindow(name="requests", used=9, limit=None)
        assert window.pressure is None

    def test_pressure_returns_none_when_limit_not_positive(self) -> None:
        window = ProviderLimitWindow(name="requests", used=9, limit=0)
        assert window.pressure is None


class TestRecoverableProviderWedge:
    def test_interrupt_timeout_is_recoverable_wedge(self) -> None:
        exc = ProviderInterruptTimeout("interrupt timed out")
        assert is_recoverable_provider_wedge(exc) is True

    def test_unrelated_exception_is_not_recoverable_wedge(self) -> None:
        assert is_recoverable_provider_wedge(RuntimeError("boom")) is False


class TestProviderLimitSnapshot:
    def test_closest_to_exhaustion_picks_highest_pressure(self) -> None:
        low = ProviderLimitWindow(name="tokens", used=20, limit=100)
        high = ProviderLimitWindow(name="requests", used=95, limit=100)
        snapshot = ProviderLimitSnapshot(
            provider=ProviderID.CLAUDE_CODE, windows=(low, high)
        )
        assert snapshot.closest_to_exhaustion() is high

    def test_closest_to_exhaustion_falls_back_to_first_window(self) -> None:
        first = ProviderLimitWindow(
            name="requests",
            resets_at=datetime(2026, 4, 16, tzinfo=UTC),
        )
        second = ProviderLimitWindow(name="tokens")
        snapshot = ProviderLimitSnapshot(
            provider=ProviderID.COPILOT_CLI,
            windows=(first, second),
        )
        assert snapshot.closest_to_exhaustion() is first

    def test_closest_to_exhaustion_returns_none_for_empty_snapshot(self) -> None:
        snapshot = ProviderLimitSnapshot(provider=ProviderID.CODEX)
        assert snapshot.closest_to_exhaustion() is None


class TestProviderPressureStatus:
    def test_from_snapshot_uses_closest_window(self) -> None:
        low = ProviderLimitWindow(name="tokens", used=20, limit=100)
        high = ProviderLimitWindow(
            name="requests",
            used=96,
            limit=100,
            resets_at=datetime(2026, 4, 16, 7, 0, tzinfo=UTC),
        )
        status = ProviderPressureStatus.from_snapshot(
            ProviderLimitSnapshot(
                provider=ProviderID.CLAUDE_CODE,
                windows=(low, high),
            )
        )
        assert status.provider is ProviderID.CLAUDE_CODE
        assert status.window_name == "requests"
        assert status.pressure == 0.96
        assert status.resets_at == datetime(2026, 4, 16, 7, 0, tzinfo=UTC)

    def test_level_is_warning_at_ninety_percent(self) -> None:
        status = ProviderPressureStatus(
            provider=ProviderID.CLAUDE_CODE,
            pressure=0.9,
        )
        assert status.level == "warning"
        assert status.warning is True
        assert status.paused is False

    def test_level_is_paused_at_ninety_five_percent(self) -> None:
        status = ProviderPressureStatus(
            provider=ProviderID.CLAUDE_CODE,
            pressure=0.95,
        )
        assert status.level == "paused"
        assert status.warning is False
        assert status.paused is True

    def test_level_is_unavailable_when_reason_present(self) -> None:
        status = ProviderPressureStatus(
            provider=ProviderID.COPILOT_CLI,
            pressure=0.99,
            unavailable_reason="limits unavailable",
        )
        assert status.level == "unavailable"

    def test_percent_used_rounds_to_nearest_whole_percent(self) -> None:
        status = ProviderPressureStatus(provider=ProviderID.CLAUDE_CODE, pressure=0.946)
        assert status.percent_used == 95

    def test_percent_used_is_none_when_pressure_unknown(self) -> None:
        status = ProviderPressureStatus(provider=ProviderID.CLAUDE_CODE)
        assert status.percent_used is None

    def test_level_is_ok_below_warning_threshold(self) -> None:
        status = ProviderPressureStatus(
            provider=ProviderID.CLAUDE_CODE,
            pressure=0.42,
        )
        assert status.level == "ok"


class TestProviderModel:
    def test_formats_and_compares_to_string(self) -> None:
        model = ProviderModel("gpt-5.4", "high")
        assert str(model) == "gpt-5.4"
        assert model == "gpt-5.4"

    def test_hash_and_model_equality_include_effort(self) -> None:
        model = ProviderModel("gpt-5.4", "high")
        same = ProviderModel("gpt-5.4", "high")
        different = ProviderModel("gpt-5.4", "medium")
        assert model == same
        assert hash(model) == hash(same)
        assert model != different

    def test_xhigh_effort_supported_for_codex(self) -> None:
        model = ProviderModel("gpt-5.5", "xhigh")
        assert model.efforts == ("xhigh",)

    def test_comparison_to_unrelated_type_is_false(self) -> None:
        assert ProviderModel("gpt-5.4") != object()


class TestProviderPalette:
    """Provider-specific color palette + palette_for lookup + contrast audit."""

    def test_palette_for_claude_code(self) -> None:
        from fido.provider import ProviderID, palette_for

        palette = palette_for(ProviderID.CLAUDE_CODE)
        assert palette is not None
        assert palette.dim_bg == (60, 30, 5)
        assert palette.bright_fg == (255, 160, 60)

    def test_palette_for_copilot_cli(self) -> None:
        from fido.provider import ProviderID, palette_for

        palette = palette_for(ProviderID.COPILOT_CLI)
        assert palette is not None
        assert palette.dim_bg == (40, 20, 60)
        assert palette.bright_fg == (180, 130, 255)

    def test_palette_for_codex(self) -> None:
        from fido.provider import ProviderID, palette_for

        palette = palette_for(ProviderID.CODEX)
        assert palette is not None
        assert palette.dim_bg == (8, 54, 43)
        assert palette.bright_fg == (72, 220, 160)

    @staticmethod
    def _relative_luminance(rgb: tuple[int, int, int]) -> float:
        """WCAG relative luminance for an sRGB triple."""

        def channel(value: int) -> float:
            c = value / 255.0
            return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

        r, g, b = rgb
        return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)

    @classmethod
    def _contrast_ratio(cls, a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
        la = cls._relative_luminance(a)
        lb = cls._relative_luminance(b)
        lighter, darker = (la, lb) if la >= lb else (lb, la)
        return (lighter + 0.05) / (darker + 0.05)

    def test_every_palette_clears_wcag_aa_vs_white(self) -> None:
        """Dim-bg tints must keep white foreground text readable (≥4.5:1).

        Prevents silent regressions when someone adds a new provider or
        tweaks colors: the tint's dim_bg must preserve contrast with the
        most common fg color used in status lines (white-ish).
        """
        from fido.provider import PROVIDER_PALETTES

        white = (255, 255, 255)
        failures: list[str] = []
        for pid, palette in PROVIDER_PALETTES.items():
            ratio = self._contrast_ratio(palette.dim_bg, white)
            if ratio < 4.5:
                failures.append(
                    f"{pid}: dim_bg={palette.dim_bg} vs white → {ratio:.2f}:1 (need ≥4.5)"
                )
        assert not failures, "\n".join(failures)

    def test_bright_fg_clears_wcag_aa_vs_black(self) -> None:
        """Bright fg on a typical dark-terminal bg must stay readable (≥4.5:1).

        Light-terminal users get worse contrast — they should opt out
        with NO_COLOR.  This test guards the dark-terminal happy path.
        """
        from fido.provider import PROVIDER_PALETTES

        black = (0, 0, 0)
        failures: list[str] = []
        for pid, palette in PROVIDER_PALETTES.items():
            ratio = self._contrast_ratio(palette.bright_fg, black)
            if ratio < 4.5:
                failures.append(
                    f"{pid}: bright_fg={palette.bright_fg} vs black → {ratio:.2f}:1"
                )
        assert not failures, "\n".join(failures)


class TestSafeVoiceTurn:
    """safe_voice_turn: retry_on_preempt + raise-on-empty contract."""

    def _agent(self, return_value: str = "ok") -> MagicMock:
        agent = MagicMock()
        agent.run_turn.return_value = return_value
        agent.voice_model = "claude-opus-4-6"
        return agent

    def test_returns_result_on_non_empty_output(self) -> None:
        agent = self._agent("Hello!")
        result = safe_voice_turn(agent, "prompt", model=agent.voice_model)
        assert result == "Hello!"

    def test_always_passes_retry_on_preempt(self) -> None:
        agent = self._agent("reply")
        safe_voice_turn(agent, "prompt", model=agent.voice_model)
        _, kwargs = agent.run_turn.call_args
        assert kwargs.get("retry_on_preempt") is True

    def test_raises_on_empty_string(self) -> None:
        agent = self._agent("")
        with pytest.raises(ValueError, match="run_turn returned empty"):
            safe_voice_turn(agent, "prompt")

    def test_error_includes_log_prefix(self) -> None:
        agent = self._agent("")
        with pytest.raises(ValueError, match="my_caller"):
            safe_voice_turn(agent, "prompt", log_prefix="my_caller")

    def test_passes_model_and_system_prompt(self) -> None:
        from fido.provider import ProviderModel

        agent = self._agent("ok")
        model = ProviderModel("claude-haiku-4-5")
        safe_voice_turn(agent, "prompt", model=model, system_prompt="be brief")
        _, kwargs = agent.run_turn.call_args
        assert kwargs["model"] == model
        assert kwargs["system_prompt"] == "be brief"
