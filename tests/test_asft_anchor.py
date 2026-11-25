"""ASFT's KL anchor: full-vocabulary KL(reference || policy) summed over supervised positions."""

import torch
import torch.nn.functional as F

from sgl.training.losses import IGNORE_INDEX, reference_kl


def _direct(logits, reference, labels):
    # upstream ASFT: F.kl_div(log_softmax(policy), softmax(reference)).sum(-1) on valid positions
    kl = F.kl_div(F.log_softmax(logits[:, :-1], -1), F.softmax(reference[:, :-1], -1), reduction="none").sum(-1)
    return kl[labels[:, 1:] != IGNORE_INDEX].sum()


def test_reference_kl_matches_upstream_formula_and_gradient():
    torch.manual_seed(0)
    logits = torch.randn(2, 7, 11, requires_grad=True)
    reference = torch.randn(2, 7, 11)
    labels = torch.randint(0, 11, (2, 7))
    labels[0, :3] = IGNORE_INDEX
    ours = reference_kl(logits, reference, labels, chunk_size=3)
    expected = _direct(logits, reference, labels)
    assert torch.allclose(ours, expected, atol=1e-5)
    (grad_ours,) = torch.autograd.grad(ours, logits)
    (grad_expected,) = torch.autograd.grad(_direct(logits, reference, labels), logits)
    assert torch.allclose(grad_ours, grad_expected, atol=1e-6)
    assert grad_ours[0, :2].abs().sum() == 0  # masked positions get no gradient


def test_reference_kl_is_zero_for_identical_models():
    logits = torch.randn(1, 5, 9)
    labels = torch.zeros(1, 5, dtype=torch.long)
    assert reference_kl(logits, logits.clone(), labels).abs() < 1e-6
