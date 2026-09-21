"""실제 GPU 네 장에서 NF4 연산과 역전파가 가능한지 확인한다. QA 결과가 아니다."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gaim.environment import configure
devices = configure()

import torch
import bitsandbytes as bnb
import transformers
import peft

print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda,
                  "transformers": transformers.__version__, "peft": peft.__version__,
                  "bitsandbytes": bnb.__version__, "visible_gpu_count": torch.cuda.device_count()}), flush=True)
assert torch.cuda.device_count() == 4
for index in range(4):
    with torch.cuda.device(index):
        device = f"cuda:{index}"
        layer = bnb.nn.Linear4bit(64, 32, bias=False, compute_dtype=torch.bfloat16, quant_type="nf4").to(device)
        x = torch.randn(2, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
        loss = layer(x).square().mean()
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
        print(json.dumps({"logical_gpu": index, "physical_gpu": devices[index],
                          "name": torch.cuda.get_device_name(index), "nf4_forward_backward": "passed"}), flush=True)
        del x, loss, layer
        torch.cuda.empty_cache()
