# error_handler3.py
"""
Centralized error handling for performance features.

How to use:
- Import ErrorHandler and custom exceptions in modules that perform backtests, demo trades, or market calls.
- Wrap risky operations in try/except and call `await ErrorHandler.handle_error(event, exc, bot=self)`.

This module provides:
- Custom exception classes for domain-specific errors.
- ErrorHandler.handle_error: maps exceptions -> user-friendly messages, logs, and optional telemetry.
"""

import asyncio
import logging
import traceback
from typing import Optional, Any, Dict

# Optional telemetry - only imported/used if configured by env (keeps runtime cheap)
try:
    import sentry_sdk  # optional
    SENTRY_AVAILABLE = True
except Exception:
    SENTRY_AVAILABLE = False

logger = logging.getLogger("performance.error_handler")
logger.setLevel(logging.INFO)

# -----------------------
# Domain-specific exceptions
# -----------------------
class PerformanceTrackingError(Exception):
    """Base exception for all performance-tracking operations."""
    pass

class MarketAPIError(PerformanceTrackingError):
    """Errors from market data providers (network, rate-limit, malformed response)."""
    def __init__(self, message: str, provider: Optional[str] = None, code: Optional[int] = None):
        super().__init__(message)
        self.provider = provider
        self.code = code

class BacktestError(PerformanceTrackingError):
    """Raised when a backtest fails or cannot be completed (e.g., not enough history)."""
    pass

class DemoTradingError(PerformanceTrackingError):
    """Raised for demo trading errors (e.g., invalid exit rule, position sizing error)."""
    pass

class PersistenceError(PerformanceTrackingError):
    """Raised when persisting or loading user/trade data fails."""
    pass

class ValidationError(PerformanceTrackingError):
    """Raised when user supplied configuration is invalid."""
    pass

# -----------------------
# ErrorHandler
# -----------------------
class ErrorHandler:
    """
    Centralized async error handler.

    Primary function:
        await ErrorHandler.handle_error(event, exc, bot=bot_instance, extra={"key": "val"})

    - event: an event-like object the bot receives (should have .sender_id and either .respond / .reply).
      If you don't have an `event`, pass user_id via `user_id` kwarg and `bot` to send messages.
    - exc: the exception instance caught.
    - bot: optional reference to Telegram bot object (used to send messages if event is not available).
    - extra: optional dict with debugging metadata (e.g., config, job_id).
    """

    @staticmethod
    async def handle_error(event: Optional[Any], exc: Exception, *,
                           bot: Optional[Any] = None,
                           user_id: Optional[int] = None,
                           extra: Optional[Dict[str, Any]] = None,
                           notify_user_for_internal_errors: bool = False) -> None:
        """
        Process `exc`, log, optionally send telemetry, and send a user-friendly message.

        - event: the incoming telegram event object (can be None).
        - bot: fallback interface for sending messages if event not provided.
        - user_id: fallback user id if event not provided.
        - extra: developer-supplied metadata to include in logs/telemetry.
        """

        # Build a canonical user id
        uid = None
        try:
            if event is not None:
                uid = getattr(event, "sender_id", None) or getattr(event, "from_user", None) and getattr(event.from_user, "id", None)
            if user_id:
                uid = user_id
        except Exception:
            uid = user_id

        # Short exception info
        exc_type = type(exc).__name__
        exc_msg = str(exc)
        tb_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))

        # Structured logging
        log_context = {"user_id": uid, "exception": exc_type, "message": exc_msg}
        if extra:
            log_context["extra"] = extra
        logger.error("Handled error: %s - %s", exc_type, exc_msg, exc_info=exc)

        # (Optional) Send to Sentry or other telemetry if available
        if SENTRY_AVAILABLE and sentry_sdk.Hub.current.client:
            try:
                with sentry_sdk.push_scope() as scope:
                    if uid:
                        scope.set_user({"id": uid})
                    if extra:
                        for k, v in (extra.items() if isinstance(extra, dict) else []):
                            scope.set_extra(k, v)
                    sentry_sdk.capture_exception(exc)
            except Exception as sentry_exc:
                logger.debug("Sentry capture failed: %s", sentry_exc)

        # Decide user-friendly message based on exception type or message heuristics
        user_message = None

        # Market API issues
        if isinstance(exc, MarketAPIError):
            msg_lower = exc_msg.lower()
            if "rate limit" in msg_lower or (hasattr(exc, "code") and exc.code in (429,)):
                user_message = (
                    "⚠️ **Market Data Rate Limit**\n\n"
                    "We're currently making many requests to the market data provider and have hit a rate limit. "
                    "Please try again in a few moments. If this keeps happening, try switching providers or reducing the query frequency."
                )
            else:
                provider = getattr(exc, "provider", None)
                provider_note = f" (provider: {provider})" if provider else ""
                user_message = (
                    f"❌ **Market Data Error**\n\n"
                    f"An error occurred while fetching market data{provider_note}: {exc_msg}\n\n"
                    "Try again in a minute or choose a different provider in your settings."
                )

        elif isinstance(exc, BacktestError):
            user_message = (
                "❌ **Backtest Failed**\n\n"
                f"Backtest could not be completed: {exc_msg}\n\n"
                "Check your backtest configuration (time window, symbol validity, and liquidity filters) and try again."
            )

        elif isinstance(exc, DemoTradingError):
            user_message = (
                "❌ **Demo Trading Error**\n\n"
                f"A problem occurred while running your demo trading session: {exc_msg}\n\n"
                "The session may have been stopped. Please review your demo settings and try again."
            )

        elif isinstance(exc, PersistenceError):
            user_message = (
                "❌ **Data Persistence Error**\n\n"
                "We encountered a problem saving or loading your data. This is likely a temporary issue. "
                "If it persists, please contact support with a short description of what you were doing."
            )

        elif isinstance(exc, ValidationError):
            # validation errors are expected friction — give clear guidance
            user_message = (
                "⚠️ **Invalid Configuration**\n\n"
                f"{exc_msg}\n\n"
                "Please correct the highlighted fields and try again."
            )

        else:
            # Unknown/unexpected errors
            logger.exception("Unhandled exception for user %s: %s", uid, exc)
            # Optionally show a friendly message but avoid leaking internals
            if notify_user_for_internal_errors:
                user_message = (
                    "❌ **An unexpected error occurred**\n\n"
                    "I've logged the details and the engineering team has been notified. "
                    "Please try again. If the problem persists, contact support and include the action you attempted."
                )
            else:
                # Minimal message to the user to avoid confusing internal stacktraces
                user_message = (
                    "❌ **An unexpected error occurred**\n\n"
                    "Something went wrong while processing your request. The problem has been logged."
                )

        # Send the message to the user (best-effort)
        try:
            if event is not None:
                # Prefer event.respond or event.reply
                if hasattr(event, "respond"):
                    await _safely_await(event.respond(user_message))
                elif hasattr(event, "reply"):
                    await _safely_await(event.reply(user_message))
                else:
                    # Generic fallback: try calling .bot.send_message if available
                    bot_obj = getattr(event, "bot", None) or bot
                    if bot_obj and uid:
                        await _safely_await(_send_via_bot(bot_obj, uid, user_message))
            elif bot is not None and uid:
                await _safely_await(_send_via_bot(bot, uid, user_message))
        except Exception as send_exc:
            # If we fail to inform the user, log the failure but don't raise
            logger.error("Failed to send error message to user %s: %s", uid, send_exc, exc_info=True)

        # End: nothing returned (fire-and-forget). The caller may choose to re-raise or swallow.

# -----------------------
# Helper utilities
# -----------------------
async def _safely_await(maybe_coro):
    """Await a coroutine or return a value if not awaitable."""
    if asyncio.iscoroutine(maybe_coro):
        return await maybe_coro
    return maybe_coro

async def _send_via_bot(bot_obj: Any, user_id: int, message: str):
    """
    Generic send wrapper for different bot libraries:
    - Telethon: bot.send_message(user_id, message)
    - python-telegram-bot: bot.send_message(chat_id=user_id, text=message)
    - Aiogram: bot.send_message(user_id, message)
    """
    try:
        # Try common method names, one by one
        if hasattr(bot_obj, "send_message"):
            # telethon and many bots use send_message(chat, text)
            await _safely_await(bot_obj.send_message(user_id, message))
            return
        if hasattr(bot_obj, "send"):
            await _safely_await(bot_obj.send(user_id, message))
            return
        # Last-resort: try attribute 'api' or 'client'
        if hasattr(bot_obj, "client") and hasattr(bot_obj.client, "send_message"):
            await _safely_await(bot_obj.client.send_message(user_id, message))
            return
    except Exception as e:
        logger.debug("Generic bot send failed: %s", e)
        raise
