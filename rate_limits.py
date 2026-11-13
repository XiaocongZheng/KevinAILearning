import re
from typing import Optional


def exception_chain_text(error: BaseException) -> str:
    messages = []
    current: Optional[BaseException] = error
    while current is not None:
        messages.append(str(current))
        current = current.__cause__
    return " ".join(messages).lower()


def retry_after_seconds(error_text: str) -> Optional[float]:
    match = re.search(r"retry in ([0-9hms.]+)", error_text.lower())
    if match is None:
        return None

    multipliers = {"h": 3600, "m": 60, "s": 1}
    parts = re.findall(r"([0-9]+(?:\.[0-9]+)?)([hms])", match.group(1))
    if not parts:
        return None
    return sum(float(amount) * multipliers[unit] for amount, unit in parts)


def is_rate_limit_error(error: BaseException) -> bool:
    error_text = exception_chain_text(error)
    if "per day" in error_text or "perday" in error_text:
        return False

    return any(
        marker in error_text
        for marker in (
            "429",
            "resource_exhausted",
            "too many requests",
            "rate limit",
            "requests per minute",
            "requestsperminute",
            "quota exceeded",
        )
    )
