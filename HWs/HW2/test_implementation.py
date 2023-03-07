import torch
from translit.py import MultiHeadAttention
import unittest

def test_positional_encoding():
        seq = torch.zeros([2, 4, 3])
        pe = PositionalEncoding(3)
        seq = pe.eval()(seq)
        self.assertTrue(abs(seq[0][1][0] - 0.8415) < 1e-4)

def test_multi_head_attention():
    multi_head_attention = MultiHeadAttention(8, 128)
    q = torch.randn(64, 4, 128)
    k = torch.randn(64, 4, 128)
    v = torch.randn(64, 4, 128)
    y = multi_head_attention(q, k, v)
    assert list(y.shape) == [64, 4, 128]
    assert list(multi_head_attention.attn.shape) == [64, 64, 4, 8]
