import subprocess
import pytest
import torch
from pathlib import Path

from model import CausalSelfAttention, GPTConfig, ShrinkCausalSelfAttention, GrowCausalSelfAttention, sliding_windows, add_shrinked_grown

BASE_DIR = Path(__file__).parent

def assert_golden(name, value):
    path = BASE_DIR / "tests" / f"{name}.txt"
    with path.open("w") as f:
        f.write(str(value))
    if subprocess.check_output(["git", "status", "--porcelain", path.absolute()]):
        subprocess.call(["git", "diff", path.absolute()])
        assert False, f"differences in {path.name}, see above"

def prepend_dim_arange(x, n):
    a = torch.arange(n) + 1
    for _ in range(len(x.shape)):
        a = a.unsqueeze(-1)
    return a * x.unsqueeze(0)

def test_prepend_dim_arange():
    torch.testing.assert_close(prepend_dim_arange(torch.arange(3), 2), torch.tensor([[0, 1, 2], [0, 2, 4]]))

def test_sliding_windows():
    x = torch.tensor([[[[1, 1], [1, 2], [1, 3]], [[2, 1], [2, 2], [2, 3]]]])
    assert_golden("sliding_windows_234_3", sliding_windows(x, 3))
    assert_golden("sliding_windows_234_4", sliding_windows(x, 4))
    assert_golden("sliding_windows_234_1", sliding_windows(x, 1))
    assert_golden("sliding_windows_234_0", sliding_windows(x, 0))
    assert_golden("sliding_windows_234_heads", sliding_windows(torch.concat((x, 2 * x), dim=1), 2))

def test_add_shrink_grow():
    a = torch.tensor([[[1], [2], [3]]])
    b = torch.zeros_like(a)
    torch.testing.assert_close(add_shrinked_grown(a, b), a)
    torch.testing.assert_close(add_shrinked_grown(b, a), a)
    torch.testing.assert_close(add_shrinked_grown(a, a), 2 * a)
    # 1 2 3 1 2 3[:,::2] is:
    # 1 3 2
    #+1 2 3
    #=2 5 5
    aa_a = torch.tensor([[[2], [5], [5]]])
    torch.testing.assert_close(add_shrinked_grown(torch.concat((a, a), dim=1), a), aa_a)
    # 1 1 2 2 3 3
    #+1 2 3 1 2 3
    #=2 3 5 3 5 6
    a_aa = torch.tensor([[[2], [3], [5], [3], [5], [6]]])
    torch.testing.assert_close(add_shrinked_grown(a, torch.concat((a, a), dim=1)), a_aa)

    a = torch.concat((a, a), dim=1)
    b = torch.zeros_like(a)
    aa_a = torch.concat((aa_a, aa_a), dim=1)
    a_aa = torch.concat((a_aa, a_aa), dim=1)
    torch.testing.assert_close(add_shrinked_grown(a, b), a)
    torch.testing.assert_close(add_shrinked_grown(b, a), a)
    torch.testing.assert_close(add_shrinked_grown(a, a), 2 * a)
    torch.testing.assert_close(add_shrinked_grown(torch.concat((a, a), dim=1), a), aa_a)
    torch.testing.assert_close(add_shrinked_grown(a, torch.concat((a, a), dim=1)), a_aa)

def test_add_shrink_grow_odd_sizes():
    a = torch.tensor([[[1], [2], [3], [4], [5]]])
    b = torch.tensor([[[2], [4], [6]]])
    # 1 2 3 4 5[:,::2] is:
    # 1 3 5
    #+2 4 6
    #=3 7 11
    a_b = torch.tensor([[[3], [7], [11]]])
    torch.testing.assert_close(add_shrinked_grown(a, b), a_b)
    # 2 2 4 4 6 6
    #+1 2 3 4 5
    #=3 4 7 8 11
    b_a = torch.tensor([[[3], [4], [7], [8], [11]]])
    torch.testing.assert_close(add_shrinked_grown(b, a), b_a)

@pytest.mark.parametrize("cls,c_attn_n", [
    (CausalSelfAttention, 3), 
    (ShrinkCausalSelfAttention, 3),
    (GrowCausalSelfAttention, 4)
])
def test_attention(cls, c_attn_n):
    B, T, W, H, E = 2, 5, 3, 2, 3
    C = H * E
    config = GPTConfig(
        block_size=T,
        n_embd=H * E,
        n_head=H,
        bias=False,
        dropout=0.0
    )
    
    attention = cls(config, context_size=W)
    eye = torch.eye(C, C)
    attn_weight = torch.concat(c_attn_n * (eye,), dim=0)
    assert attention.c_attn.weight.shape == attn_weight.shape
    attention.c_attn.weight.data = attn_weight
    assert attention.c_proj.weight.shape == eye.shape
    attention.c_proj.weight.data = eye
    vec = torch.arange(C, dtype=torch.float)
    seq = prepend_dim_arange(vec, T)
    batch = prepend_dim_arange(seq, B)
    attn = attention(batch)
    assert_golden(cls.__name__, attn)
