import torch

from adaptation.modules import EntropyCalculator, AcceptanceTracker, KController
from metrics.trackers import PerformanceTracker, QualityTracker
from verification.verifier import Verifier


class SpeculativeEngine:
    """
    Entropy-Aware Adaptive Speculative Decoding Engine.

    Clean architecture:
    - Draft model generates k tokens autoregressively
    - Verifier checks each token using forward_next (no cache manipulation)
    - Cache state is always correct — no rollbacks, no deepcopy, no DynamicCache tricks
    - k adapts based on entropy and acceptance rate
    """

    def __init__(
        self,
        draft_model,
        target_model,
        max_k: int = 4,
        entropy_thresholds=None,
        k_values=None,
        temperature: float = 1.0,
        acceptance_alpha: float = 0.1,
        acceptance_init: float = 0.7,
    ):
        self.draft_model = draft_model
        self.target_model = target_model
        self.temperature = temperature

        # Adaptation
        self.entropy_calc = EntropyCalculator()
        self.acceptance_tracker = AcceptanceTracker(acceptance_alpha, acceptance_init)

        if entropy_thresholds is None:
            entropy_thresholds = [1.5, 3.0, 5.0]
        if k_values is None:
            k_values = [4, 3, 2, 1]

        self.k_controller = KController(
            entropy_thresholds=entropy_thresholds,
            k_values=k_values,
            k_max=max_k,
        )

        # Verifier — token by token, no cache tricks
        self.verifier = Verifier(
            target_model=target_model,
            draft_model=draft_model,
            temperature=temperature,
        )

        # Metrics
        self.perf = PerformanceTracker()
        self.quality = QualityTracker()

        # History for plotting
        self.k_history = []
        self.acceptance_history = []

    @torch.no_grad()
    def decode(self, input_ids: torch.Tensor, max_tokens: int = 100):
        """
        Run adaptive speculative decoding.

        Args:
            input_ids: (1, seq_len) prompt tensor
            max_tokens: maximum new tokens to generate

        Returns:
            output_ids: (1, seq_len + generated) tensor
        """
        self.perf.reset()
        self.quality.reset()
        self.k_history.clear()
        self.acceptance_history.clear()

        # ── Init both models with prompt ──
        self.draft_model.reset()
        self.target_model.reset()

        draft_logits = self.draft_model.init_kv_cache(input_ids)
        self.target_model.init_kv_cache(input_ids)

        output_ids = input_ids.clone()
        self.perf.start()

        for _ in range(max_tokens):

            # ── 1. Compute entropy from draft logits ──
            entropy = self.entropy_calc.compute(draft_logits)

            # ── 2. Decide k ──
            k = self.k_controller.decide_k(
                entropy=entropy,
                acceptance_rate=self.acceptance_tracker.value,
            )
            self.k_history.append(k)

            # ── 3. k=0 fallback — vanilla target step ──
            if k == 0:
                t_logits = self.target_model.forward_next(output_ids[:, -1:])
                next_token = self.target_model.select_tokens(t_logits)
                output_ids = torch.cat([output_ids, next_token], dim=1)

                # Sync draft model
                draft_logits = self.draft_model.forward_next(
                    next_token.to(self.draft_model.device)
                )

                self.perf.record_target_forward(1)
                self.perf.record_tokens(1)
                continue

            # ── 4. Generate k draft tokens ──
            # Both models currently at position output_ids.shape[1]
            # After generate: draft model at output_ids.shape[1] + k
            # Target model still at output_ids.shape[1] (untouched)
            draft_tokens = self._generate_draft(k, output_ids[:, -1:])
            self.perf.record_draft_forward(k)

            # ── 5. Verify ──
            # Verifier runs target forward_next on each draft token
            # then runs draft forward_next on each draft token
            # On rejection: both caches rolled back to rejection point, corrected token appended
            # On full acceptance: bonus token sampled and appended
            # After verify: both caches at output_ids.shape[1] + n_accepted + 1
            n_accepted, next_token = self.verifier.verify(draft_tokens)

            self.perf.record_target_forward(1)
            self.quality.record(k, n_accepted)
            self.acceptance_tracker.update(n_accepted, k)
            self.acceptance_history.append(n_accepted / k)

            # ── 6. Commit to output_ids ──
            if n_accepted > 0:
                output_ids = torch.cat(
                    [output_ids, draft_tokens[:, :n_accepted], next_token], dim=1
                )
            else:
                output_ids = torch.cat([output_ids, next_token], dim=1)

            self.perf.record_tokens(n_accepted + 1)

            # ── 7. Get draft logits for next entropy calculation ──
            # Both caches at output_ids.shape[1] after commit
            # forward_next advances draft cache by 1 for next iteration
            draft_logits = self.draft_model.forward_next(
                next_token.to(self.draft_model.device)
            )

            # ── 8. EOS check ──
            if next_token.item() == self.target_model.tokenizer.eos_token_id:
                break

        self.perf.stop()
        return output_ids

    def _generate_draft(self, k: int, last_token: torch.Tensor) -> torch.Tensor:
        """Generate k draft tokens autoregressively."""
        tokens = []
        current = last_token.to(self.draft_model.device)

        for _ in range(k):
            logits = self.draft_model.forward_next(current)
            next_tok = self.draft_model.sample_token(logits)
            tokens.append(next_tok)
            current = next_tok

        return torch.cat(tokens, dim=1)  # (1, k)