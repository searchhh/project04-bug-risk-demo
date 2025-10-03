
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

MODEL_ID = "microsoft/codebert-base"

tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID).cuda().eval()


def encode(texts, max_len=256):
    batch = tok(texts, padding=True, truncation=True,
                max_length=max_len, return_tensors="pt")
    for k in batch:
        batch[k] = batch[k].cuda()
    with torch.no_grad():
        out = model(**batch)  # last_hidden_state [B, L, H]
        # 取 [CLS] 向量或平均池化都可，这里示例平均池化（更稳）
        mask = batch["attention_mask"].unsqueeze(-1)  # [B,L,1]
        emb = (out.last_hidden_state * mask).sum(dim=1) / \
            mask.sum(dim=1).clamp(min=1e-9)
        emb = F.normalize(emb, dim=-1)
    return emb


code_a = """def add(a,b): return a+b"""
code_b = """def sum_two(x,y): return x+y"""
code_c = """def multiply(a,b): return a*b"""

E = encode([code_a, code_b, code_c])  # [3, hidden]
sim_ab = (E[0] @ E[1]).item()
sim_ac = (E[0] @ E[2]).item()
print("cos(a,b)=", round(sim_ab, 3), "cos(a,c)=", round(sim_ac, 3))
