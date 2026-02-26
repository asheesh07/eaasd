import time


class PerformanceTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self._start = None
        self.total_time = 0.0
        self.tokens_generated = 0
        self.target_forward_calls = 0
        self.draft_forward_calls = 0

    def start(self):
        self._start = time.time()

    def stop(self):
        self.total_time = time.time() - self._start

    def record_tokens(self, n: int):
        self.tokens_generated += n

    def record_target_forward(self, n: int = 1):
        self.target_forward_calls += n

    def record_draft_forward(self, n: int = 1):
        self.draft_forward_calls += n

    def summary(self) -> dict:
        lat = (self.total_time / self.tokens_generated * 1000
               if self.tokens_generated else 0)
        tput = (self.tokens_generated / self.total_time
                if self.total_time else 0)
        return {
            "total_time_sec": round(self.total_time, 4),
            "tokens_generated": self.tokens_generated,
            "latency_per_token_ms": round(lat, 3),
            "throughput_tokens_per_sec": round(tput, 2),
            "target_forward_calls": self.target_forward_calls,
            "draft_forward_calls": self.draft_forward_calls,
        }

    @property
    def latency_per_token_ms(self):
        return (self.total_time / self.tokens_generated * 1000
                if self.tokens_generated else 0)


class QualityTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.steps = 0
        self.total_k = 0
        self.total_accepted = 0
        self.rejections = 0

    def record(self, k: int, accepted: int):
        self.steps += 1
        self.total_k += k
        self.total_accepted += accepted
        if accepted < k:
            self.rejections += 1

    def summary(self) -> dict:
        acc = self.total_accepted / self.total_k if self.total_k else 0
        avg_k = self.total_k / self.steps if self.steps else 0
        rej_rate = self.rejections / self.steps if self.steps else 0
        return {
            "total_steps": self.steps,
            "average_k": round(avg_k, 2),
            "acceptance_rate": round(acc, 4),
            "rejection_rate": round(rej_rate, 2),
            "wasted_speculation": round(1 - acc, 4),
        }