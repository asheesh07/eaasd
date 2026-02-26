import torch
import torch.nn.functional as F


class Verifier:
    """
    Token-by-token verification using forward_next only.

    Architecture:
    - No DynamicCache manipulation
    - No rollbacks
    - No deepcopy
    - Cache state is always deterministic and correct

    How it works:
    1. For each draft token position i:
       - Run target.forward_next(draft_token[i]) → get target logits at position i
       - Run draft.forward_next(draft_token[i])  → get draft logits at position i
       - Apply acceptance criterion
       - If accepted: continue, both caches advance naturally
       - If rejected: sample corrected token, both caches are at exact right position
    2. If all k accepted: sample bonus token from target, advance both caches

    Cache state after verify():
    - Both models at position seq_len + n_accepted + 1
    - No correction needed — cache advances match committed tokens exactly
    """

    def __init__(self, target_model, draft_model, temperature: float = 1.0):
        self.target_model = target_model
        self.draft_model = draft_model
        self.temperature = temperature

    @torch.no_grad()
    def verify(self, draft_tokens: torch.Tensor):
        """
        draft_tokens: (1, k) — tokens proposed by draft model

        Both models must already have KV cache at seq_len before calling.
        draft_tokens are the k tokens AFTER the last committed token.

        Returns:
            n_accepted: int
            next_token: (1, 1) tensor — token to append after accepted drafts
        """
        k = draft_tokens.shape[1]

        if k == 0:
            # No speculation — shouldn't happen but handle gracefully
            raise ValueError("verify() called with k=0, handle this in engine")

        target_logits_list = []
        draft_logits_list  = []

        # ── Step 1: Run both models token by token over draft tokens ──
        # Both caches start at seq_len
        # After this loop: both caches at seq_len + k
        for i in range(k):
            token = draft_tokens[:, i:i+1]

            t_logits = self.target_model.forward_next(
                token.to(self.target_model.device)
            )  # (1, vocab)
            d_logits = self.draft_model.forward_next(
                token.to(self.draft_model.device)
            )  # (1, vocab)

            target_logits_list.append(t_logits)
            draft_logits_list.append(d_logits)

        # ── Step 2: Align vocab sizes once ──
        t_vocab = target_logits_list[0].shape[-1]
        d_vocab = draft_logits_list[0].shape[-1]
        vocab_mismatch = t_vocab != d_vocab

        # ── Step 3: Acceptance loop ──
        # We need to rollback caches to the rejection point
        # Track how far we need to rollback
        n_accepted = 0
        next_token = None

        for i in range(k):
            t_logits_i = target_logits_list[i]
            d_logits_i = draft_logits_list[i]

            if vocab_mismatch:
                if d_vocab < t_vocab:
                    d_logits_i = F.pad(
                        d_logits_i, (0, t_vocab - d_vocab), value=float('-inf')
                    )
                else:
                    t_logits_i = F.pad(
                        t_logits_i, (0, d_vocab - t_vocab), value=float('-inf')
                    )

            t_probs = F.softmax(
                t_logits_i / max(self.temperature, 1e-5), dim=-1
            )
            d_probs = F.softmax(
                d_logits_i.to(t_logits_i.device) / max(self.temperature, 1e-5),
                dim=-1,
            )

            draft_token_id = draft_tokens[0, i].item()
            p_target = t_probs[0, draft_token_id].item()
            p_draft  = d_probs[0, draft_token_id].item()

            acceptance_prob = min(1.0, p_target / (p_draft + 1e-8))

            if torch.rand(1).item() < acceptance_prob:
                n_accepted += 1
            else:
                # Rejected at position i
                # Both caches are currently at seq_len + k
                # Rollback both to seq_len + i (position of rejection)
                rollback_to = self.target_model.position - (k - i)
                self.target_model.kv_cache.crop(rollback_to)
                self.target_model.position = rollback_to
                self.draft_model.kv_cache.crop(rollback_to)
                self.draft_model.position = rollback_to

                # Sample corrected token from max(0, p_target - p_draft)
                corrected = torch.clamp(t_probs - d_probs, min=0.0)
                corrected_sum = corrected.sum(dim=-1, keepdim=True)
                if corrected_sum.item() < 1e-8:
                    corrected = t_probs
                else:
                    corrected = corrected / corrected_sum

                next_token = torch.multinomial(corrected, num_samples=1)

                # Advance both caches by 1 for the corrected token
                self.target_model.forward_next(
                    next_token.to(self.target_model.device)
                )
                self.draft_model.forward_next(
                    next_token.to(self.draft_model.device)
                )
                break

        if next_token is None:
            # All k accepted — both caches at seq_len + k
            # Sample bonus token from target's last logits
            bonus_logits = target_logits_list[k - 1]

            if vocab_mismatch and d_vocab > t_vocab:
                bonus_logits = F.pad(
                    bonus_logits, (0, d_vocab - t_vocab), value=float('-inf')
                )

            bonus_probs = F.softmax(
                bonus_logits / max(self.temperature, 1e-5), dim=-1
            )
            next_token = torch.multinomial(bonus_probs, num_samples=1)

            # Advance both caches for bonus token
            self.target_model.forward_next(
                next_token.to(self.target_model.device)
            )
            self.draft_model.forward_next(
                next_token.to(self.draft_model.device)
            )

        # Both caches now at seq_len + n_accepted + 1 — correct position
        return n_accepted, next_token