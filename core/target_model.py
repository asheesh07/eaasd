import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache


class TargetModel:
    def __init__(
        self,
        tokenizer,
        model_name: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ):
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype
        ).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.device = device
        self.dtype = dtype
        self.kv_cache = None
        self.position = 0

    @torch.no_grad()
    def init_kv_cache(self, input_ids: torch.Tensor):
        self.kv_cache = DynamicCache()
        input_ids = input_ids.to(self.device)
        outputs = self.model(
            input_ids=input_ids,
            past_key_values=self.kv_cache,
            use_cache=True,
            return_dict=True,
        )
        self.kv_cache = outputs.past_key_values
        self.position = input_ids.shape[1]
        return outputs.logits[:, -1, :]

    @torch.no_grad()
    def forward_next(self, input_ids: torch.Tensor):
        assert input_ids.shape[-1] == 1, f"Expected 1 token, got {input_ids.shape}"
        input_ids = input_ids.to(self.device)
        outputs = self.model(
            input_ids=input_ids,
            past_key_values=self.kv_cache,
            use_cache=True,
            return_dict=True,
        )
        self.kv_cache = outputs.past_key_values
        self.position += 1
        return outputs.logits[:, -1, :]

    def select_tokens(self, logits: torch.Tensor) -> torch.Tensor:
        logits = logits / max(self.temperature, 1e-5)
        probs = F.softmax(logits, dim=-1)

        if self.top_k > 0:
            top_k_probs, top_k_indices = torch.topk(probs, self.top_k, dim=-1)
            top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
            sampled = torch.multinomial(top_k_probs, num_samples=1)
            return top_k_indices.gather(-1, sampled)

        return torch.multinomial(probs, num_samples=1)

    def reset(self):
        self.kv_cache = None
        self.position = 0