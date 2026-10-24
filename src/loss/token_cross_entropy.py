from torch import nn
from torch.nn import functional as F


class TokenCrossEntropy(nn.Module):
    def forward(self, logits, labels, **unused):
        total = F.cross_entropy(
            logits.flatten(0, 1), labels.flatten(), ignore_index=-100, reduction="sum"
        )
        count = labels.ne(-100).sum()
        return {
            "loss": total / count.clamp_min(1),
            "loss_sum": total,
            "token_count": count,
        }
