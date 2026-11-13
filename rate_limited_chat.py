import threading
import time

from rate_limits import (
    exception_chain_text,
    is_rate_limit_error,
    retry_after_seconds,
)


class RateLimitedChatModel:
    def __init__(
        self,
        model: object,
        *,
        min_request_interval_seconds: float = 13,
        max_retries: int = 2,
    ) -> None:
        if min_request_interval_seconds < 0:
            raise ValueError("min_request_interval_seconds must be non-negative")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")

        self._model = model
        self._min_request_interval_seconds = min_request_interval_seconds
        self._max_retries = max_retries
        self._last_request_at = 0.0
        self._request_lock = threading.Lock()

    def invoke(self, prompt: str) -> object:
        with self._request_lock:
            for attempt in range(self._max_retries + 1):
                wait_seconds = self._min_request_interval_seconds - (
                    time.monotonic() - self._last_request_at
                )
                if wait_seconds > 0:
                    time.sleep(wait_seconds)

                self._last_request_at = time.monotonic()
                try:
                    return self._model.invoke(prompt)
                except Exception as error:
                    if (
                        not is_rate_limit_error(error)
                        or attempt == self._max_retries
                    ):
                        raise

                    retry_after = retry_after_seconds(
                        exception_chain_text(error)
                    )
                    delay = (
                        retry_after
                        if retry_after is not None
                        else self._min_request_interval_seconds
                    )
                    print(
                        "Gemini rate limit reached. "
                        f"Retrying in {delay:.1f} seconds "
                        f"({attempt + 1}/{self._max_retries})..."
                    )
                    time.sleep(delay)

        raise RuntimeError("Chat retry loop exited unexpectedly")
