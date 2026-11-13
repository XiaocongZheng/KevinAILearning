import unittest
from unittest.mock import patch

from rate_limited_chat import RateLimitedChatModel


class FakeModel:
    def __init__(self, results):
        self.results = iter(results)
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        result = next(self.results)
        if isinstance(result, BaseException):
            raise result
        return result


class RateLimitedChatTests(unittest.TestCase):
    def run_with_fake_clock(self, callback):
        clock = [100.0]
        sleeps = []

        def fake_sleep(seconds):
            sleeps.append(seconds)
            clock[0] += seconds

        with (
            patch("rate_limited_chat.time.monotonic", side_effect=lambda: clock[0]),
            patch("rate_limited_chat.time.sleep", side_effect=fake_sleep),
        ):
            result = callback()
        return result, sleeps

    def test_waits_for_server_retry_delay_then_retries(self):
        response = object()
        model = FakeModel(
            [
                RuntimeError(
                    "Quota exceeded for requests per minute. "
                    "Please retry in 41.394691255s."
                ),
                response,
            ]
        )
        limiter = RateLimitedChatModel(model)

        result, sleeps = self.run_with_fake_clock(
            lambda: limiter.invoke("question")
        )

        self.assertIs(result, response)
        self.assertEqual(model.calls, 2)
        self.assertEqual(sleeps, [41.394691255])

    def test_spaces_requests_to_stay_below_single_process_limit(self):
        model = FakeModel([object(), object()])
        limiter = RateLimitedChatModel(model)

        _, sleeps = self.run_with_fake_clock(
            lambda: (limiter.invoke("first"), limiter.invoke("second"))
        )

        self.assertEqual(sleeps, [13])

    def test_does_not_retry_non_rate_limit_errors(self):
        error = RuntimeError("invalid request")
        model = FakeModel([error])
        limiter = RateLimitedChatModel(model)

        with self.assertRaisesRegex(RuntimeError, "invalid request"):
            self.run_with_fake_clock(lambda: limiter.invoke("question"))

        self.assertEqual(model.calls, 1)

    def test_does_not_retry_daily_quota_errors(self):
        error = RuntimeError(
            "Quota exceeded for requests per day. Please retry in 12h."
        )
        model = FakeModel([error])
        limiter = RateLimitedChatModel(model)

        with self.assertRaisesRegex(RuntimeError, "per day"):
            self.run_with_fake_clock(lambda: limiter.invoke("question"))

        self.assertEqual(model.calls, 1)


if __name__ == "__main__":
    unittest.main()
