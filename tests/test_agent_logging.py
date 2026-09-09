"""Agent log files must stay silent by default and bounded when switched on.
"""

import importlib
import logging
import logging.handlers

import pytest

import settings as s


@pytest.fixture
def reloaded_settings(monkeypatch):
    """Re-import ``settings`` with a given ``AOT_LOG_LEVEL``."""

    def load(value):
        if value is None:
            monkeypatch.delenv("AOT_LOG_LEVEL", raising=False)
        else:
            monkeypatch.setenv("AOT_LOG_LEVEL", value)
        return importlib.reload(s)

    yield load
    monkeypatch.delenv("AOT_LOG_LEVEL", raising=False)
    importlib.reload(s)


def test_agents_are_silent_by_default(reloaded_settings):
    settings = reloaded_settings(None)
    assert settings.LOG_AGENT_CODE == logging.CRITICAL
    assert settings.LOG_AGENT_WRAPPER == logging.CRITICAL


@pytest.mark.parametrize(
    "value, expected",
    [("DEBUG", logging.DEBUG), ("info", logging.INFO), (" Warning ", logging.WARNING)],
)
def test_env_override_is_honoured(reloaded_settings, value, expected):
    settings = reloaded_settings(value)
    assert settings.LOG_AGENT_CODE == expected


def test_unparseable_level_falls_back_to_silence(reloaded_settings):
    """A typo must not silently restore the 2.1 GB default."""
    settings = reloaded_settings("verbose-please")
    assert settings.LOG_AGENT_CODE == logging.CRITICAL


def test_game_log_keeps_the_framework_level(reloaded_settings):
    """``logs/game.log`` is one small file per run, so it is left alone."""
    assert reloaded_settings(None).LOG_GAME == logging.INFO


def test_agent_log_handler_is_size_capped(tmp_path):
    """The handler must honour ``LOG_MAX_FILE_SIZE`` rather than grow forever."""
    log_file = tmp_path / "agent.log"
    handler = logging.handlers.RotatingFileHandler(
        str(log_file), maxBytes=1024, backupCount=1)
    logger = logging.getLogger("test_agent_log_handler_is_size_capped")
    logger.handlers = [handler]
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    for _ in range(500):
        logger.debug("x" * 200)
    handler.close()

    written = sum(p.stat().st_size for p in tmp_path.iterdir())
    assert written <= 2 * 1024 + 512, f"log grew to {written} bytes"


def test_agent_runner_uses_a_rotating_handler():
    """Guards the wiring itself: a plain FileHandler is what caused the 2.1 GB."""
    source = (s.Path(__file__).resolve().parent.parent / "agents.py").read_text()
    assert "RotatingFileHandler" in source
    assert "maxBytes=s.LOG_MAX_FILE_SIZE" in source
