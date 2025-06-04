import torch
import torch.nn as nn

################################################################################
# 1) NT-Xent Loss for SimCLR
################################################################################
class NTXentLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        N = 2 * batch_size
        mask = torch.eye(N, dtype=torch.bool)
        self.register_buffer("mask", mask)

    def forward(self, z):
        """
        z: [2*batch_size, dim]
        """
        N = 2 * self.batch_size
        # Compute similarity matrix
        sim = torch.matmul(z, z.T) / self.temperature

        # Positive pairs: (i, i + batch_size) and (i + batch_size, i)
        pos = torch.cat([
            torch.diag(sim, self.batch_size),
            torch.diag(sim, -self.batch_size)
        ]).view(2 * self.batch_size, 1)

        # Mask out self-similarities
        sim = sim.masked_fill(self.mask, -9e15)

        # Logits: [2N, 1 + (2N-1)]
        logits = torch.cat([pos, sim], dim=1)
        labels = torch.zeros(2 * self.batch_size, dtype=torch.long, device=z.device)

        loss = nn.CrossEntropyLoss()(logits, labels)
        return loss

################################################################################
# 2) Cosine Similarity (if needed)
################################################################################
def cosine_similarity(a, b, eps=1e-8):
    a_norm = a / (a.norm(dim=1, keepdim=True) + eps)
    b_norm = b / (b.norm(dim=1, keepdim=True) + eps)
    return torch.mm(a_norm, b_norm.T)
