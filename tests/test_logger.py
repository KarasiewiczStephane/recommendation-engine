"""Tests for the structured logging module."""

import logging

from src.utils.logger import get_logger


class TestGetLogger:
    """Tests for the get_logger function."""

    def test_returns_logger_instance(self) -> None:
        """get_logger returns a Logger object."""
        logger = get_logger("test")
        assert isinstance(logger, logging.Logger)

    def test_logger_has_handler(self) -> None:
        """Logger is created with at least one handler."""
        logger = get_logger("test_handler")
        assert len(logger.handlers) >= 1

    def test_default_level_is_info(self) -> None:
        """Default logging level is INFO."""
        logger = get_logger("test_level")
        assert logger.level == logging.INFO

    def test_custom_level(self) -> None:
        """Logger respects a custom level argument."""
        logger = get_logger("test_custom_level", level=logging.DEBUG)
        assert logger.level == logging.DEBUG

    def test_no_duplicate_handlers(self) -> None:
        """Calling get_logger twice does not add duplicate handlers."""
        logger1 = get_logger("test_no_dup")
        handler_count = len(logger1.handlers)
        logger2 = get_logger("test_no_dup")
        assert len(logger2.handlers) == handler_count

    def test_formatter_output(self, capsys: object) -> None:
        """Logger output contains expected format elements."""
        logger = get_logger("test_format")
        logger.info("test message")
        captured = capsys.readouterr()
        assert "test_format" in captured.out
        assert "INFO" in captured.out
        assert "test message" in captured.out
