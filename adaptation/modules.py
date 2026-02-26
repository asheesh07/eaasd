import torch
import torch.nn.functional as F


class EntropyCalculator:
    @torch.no_grad()
    def compute(self, logits: torch.Tensor) -> float:
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        entropy = -(probs * log_probs).sum(dim=-1)
        entropy = torch.nan_to_num(entropy, nan=0.0, posinf=0.0)
        return entropy.item()


class AcceptanceTracker:
    def __init__(self, alpha: float = 0.1, initial_value: float = 0.7):
        self.alpha = alpha
        self.value = initial_value

    def update(self, accepted: int, k: int):
        if k <= 0:
            return
        rate = accepted / k
        self.value = (1 - self.alpha) * self.value + self.alpha * rate


class KController:
    def __init__(
        self,
        entropy_thresholds,
        k_values,
        k_max: int,
        acceptance_min: float = 0.4,
        acceptance_max: float = 0.8,
    ):
        assert len(k_values) == len(entropy_thresholds) + 1
        self.entropy_thresholds = list(entropy_thresholds)
        self.k_values = k_values
        self.k_max = k_max
        self.acceptance_min = acceptance_min
        self.acceptance_max = acceptance_max

    def _entropy_to_k(self, entropy: float) -> int:
        for i, threshold in enumerate(self.entropy_thresholds):
            if entropy < threshold:
                return self.k_values[i]
        return self.k_values[-1]

    def decide_k(self, entropy: float, acceptance_rate: float) -> int:
        k = self._entropy_to_k(entropy)

        # Acceptance feedback — linear not exponential
        if acceptance_rate < self.acceptance_min:
            k = max(1, k - 1)
        elif acceptance_rate > self.acceptance_max:
            k = min(self.k_max, k + 1)

        return k