import time
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

from core.draft_model import DraftModel
from core.target_model import TargetModel
from core.engine import SpeculativeEngine


def fair_vanilla_baseline(target, tokenizer, input_ids, max_tokens=50, device="cuda"):
    """Vanilla baseline using same forward_next + select_tokens as engine."""
    target.reset()
    target.init_kv_cache(input_ids)

    tokens = []
    last_token = input_ids[:, -1:]

    torch.cuda.synchronize()
    start = time.time()

    with torch.no_grad():
        for _ in range(max_tokens):
            logits = target.forward_next(last_token)
            next_token = target.select_tokens(logits)
            tokens.append(next_token.item())
            last_token = next_token
            if next_token.item() == tokenizer.eos_token_id:
                break

    torch.cuda.synchronize()
    elapsed = time.time() - start

    latency = elapsed / len(tokens) * 1000
    throughput = len(tokens) / elapsed

    print("=== FAIR VANILLA BASELINE ===")
    print(f"Total time (s): {elapsed:.4f}")
    print(f"Tokens generated: {len(tokens)}")
    print(f"Latency per token (ms): {latency:.2f}")
    print(f"Throughput (tok/s): {throughput:.2f}")
    print(f"Output: {tokenizer.decode(tokens, skip_special_tokens=True)}")

    return latency


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Models ──
    MODEL_DRAFT  = "Qwen/Qwen2.5-0.5B-Instruct"
    MODEL_TARGET = "Qwen/Qwen2.5-1.5B-Instruct"

    print(f"Loading tokenizer from {MODEL_TARGET}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_TARGET)

    print(f"Loading draft model: {MODEL_DRAFT}")
    draft = DraftModel(
        tokenizer=tokenizer,
        model_name=MODEL_DRAFT,
        device=device,
        dtype=torch.float16,
        temperature=0.7,
        top_k=20,
    )

    print(f"Loading target model: {MODEL_TARGET}")
    target = TargetModel(
        tokenizer=tokenizer,
        model_name=MODEL_TARGET,
        device=device,
        dtype=torch.float16,
        temperature=1.0,
    )

    # ── Engine ──
    engine = SpeculativeEngine(
        draft_model=draft,
        target_model=target,
        max_k=4,
        entropy_thresholds=[1.5, 3.0, 5.0],
        k_values=[4, 3, 2, 1],
        temperature=1.0,
    )

    # ── Prompt ──
    prompt = "The theory of evolution explains"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    MAX_TOKENS = 50

    # ── Fair Vanilla Baseline ──
    vanilla_latency = fair_vanilla_baseline(
        target, tokenizer, input_ids, MAX_TOKENS, device
    )

    # ── Speculative Engine ──
    print("\nRunning speculative engine...")
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    torch.cuda.synchronize()
    output_ids = engine.decode(input_ids, max_tokens=MAX_TOKENS)
    torch.cuda.synchronize()

    spec_latency = engine.perf.latency_per_token_ms

    print("\n=== SPECULATIVE ENGINE ===")
    print(engine.perf.summary())
    print(engine.quality.summary())
    print(f"Output: {tokenizer.decode(output_ids[0], skip_special_tokens=True)}")
    print(f"\nSpeedup: {vanilla_latency / spec_latency:.2f}x")

    # ── Plots ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].bar(["Vanilla", "Speculative"], [vanilla_latency, spec_latency])
    axes[0].set_ylabel("Latency per Token (ms)")
    axes[0].set_title("Latency Comparison")

    axes[1].plot(engine.k_history)
    axes[1].set_xlabel("Decoding Step")
    axes[1].set_ylabel("k")
    axes[1].set_title("Adaptive Speculation Depth (k)")

    axes[2].plot(engine.acceptance_history)
    axes[2].set_xlabel("Decoding Step")
    axes[2].set_ylabel("Acceptance Rate")
    axes[2].set_title("Acceptance Rate over Time")
    axes[2].set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig("results.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()