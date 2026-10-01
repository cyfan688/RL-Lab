import torch
from model import build_transformer


model = build_transformer(
    src_vocab_size=40, tgt_vocab_size=30,
    src_seq_len=6, tgt_seq_len=5,
    d_model=16, N=1, h=4, d_ff=32, dropout=0.0
)

src = torch.tensor([[1, 2, 3, 0, 0, 0],
                    [4, 5, 6, 7, 0, 0]])
tgt = torch.tensor([[1, 8, 9, 0, 0],
                    [1, 6, 7, 8, 0]])

src_mask = (src != 0)[:, None, None, :]
causal = torch.tril(torch.ones(5, 5, dtype=torch.bool))
tgt_mask = (tgt != 0)[:, None, None, :] & causal[None, None, :, :]

encoded = model.encode(src, src_mask)
decoded = model.decode(encoded, src_mask, tgt, tgt_mask)
logits = model.project(decoded)

if __name__ == "__main__":
    print("Encoded shape:", encoded.shape)
    print("Decoded shape:", decoded.shape)
    print("Logits shape:", logits.shape)