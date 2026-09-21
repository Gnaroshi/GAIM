"""로컬 Hugging Face 모델: 각 GPU에 같은 모델 한 개를 올려 두 역할에 사용."""
from __future__ import annotations

import hashlib
import time


def call_seed(seed: int, *parts: object) -> int:
    """실행 순서나 worker 수가 바뀌어도 각 요청의 난수 seed를 유지한다."""
    payload = ":".join(map(str, (seed, *parts))).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") % (2**31)


class LocalModel:
    def __init__(self, model: str, revision: str, device: int, max_input_tokens: int,
                 adapter_path: str | None = None):
        from .environment import configure
        configure()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = f"cuda:{device}"
        torch.cuda.set_device(device)
        torch.set_num_threads(2)
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model, revision=revision, local_files_only=True,
            torch_dtype=torch.bfloat16, device_map={"": self.device},
            attn_implementation="sdpa",
        ).eval()
        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path).eval()
        self.max_input_tokens = max_input_tokens

    def generate(self, messages: list[dict], *, seed: int, temperature: float,
                 max_new_tokens: int) -> dict:
        torch = self.torch
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        length = inputs.input_ids.shape[1]
        # 원문이 잘리면 다른 문제가 되므로 자동 truncate하지 않고 오류로 남긴다.
        if length > self.max_input_tokens:
            raise ValueError(f"Input is {length} tokens; limit={self.max_input_tokens}. No truncation performed.")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        kwargs = dict(max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                      pad_token_id=self.tokenizer.eos_token_id, use_cache=True)
        if temperature > 0:
            kwargs.update(temperature=temperature, top_p=0.95, top_k=20)
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(**inputs, **kwargs)
        torch.cuda.synchronize(self.device)
        generated = output[0, length:]
        return {
            "raw": self.tokenizer.decode(generated, skip_special_tokens=True),
            "input_tokens": length, "output_tokens": len(generated),
            "latency_s": time.perf_counter() - started,
            "seed": seed,
            "hit_token_limit": len(generated) >= max_new_tokens,
        }
