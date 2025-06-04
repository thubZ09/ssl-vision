import torch
import torch.nn as nn
import torchvision.models as models
import copy

################################################################################
# 1) ResNetEncoder: backbone + projection head (for SimCLR / MoCo / BYOL / DINO)
################################################################################
class ResNetEncoder(nn.Module):
    def __init__(self, base='resnet18', out_dim=128):
        """
        base: 'resnet18' or 'resnet50'
        out_dim: dimension of projection head output
        """
        super().__init__()
        if base == 'resnet18':
            self.backbone = models.resnet18(pretrained=False)
            feat_dim = 512
        elif base == 'resnet50':
            self.backbone = models.resnet50(pretrained=False)
            feat_dim = 2048
        else:
            raise ValueError("Unsupported backbone")

        # Remove final classifier
        self.backbone.fc = nn.Identity()

        # Projection head: MLP (feat_dim → feat_dim → out_dim)
        self.projection_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, out_dim),
        )

    def forward(self, x):
        """
        Returns both:
         - h: the representation before projection (for linear eval)
         - z: the projection used in the SSL loss
        """
        h = self.backbone(x)         # shape: [B, feat_dim]
        z = self.projection_head(h)  # shape: [B, out_dim]
        return h, z

################################################################################
# 2) MoCo Wrapper
################################################################################
class MoCo(nn.Module):
    """
    MoCo‐v2 style wrapper. 
    - encoder_q: query encoder (updated by backprop)
    - encoder_k: key encoder (momentum update)
    - queue: dictionary of negative keys (normalized)
    """
    def __init__(self,
                 base_encoder,
                 dim=128,
                 K=4096,
                 m=0.99,
                 T=0.2):
        super().__init__()
        self.K = K
        self.m = m
        self.T = T

        # Query encoder
        self.encoder_q = base_encoder(out_dim=dim)
        # Key encoder
        self.encoder_k = base_encoder(out_dim=dim)

        # Initialize key encoder weights to match query
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                     self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False  # no gradients for key encoder

        # Create the queue (dim × K) and pointer
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update: θ_k ← m θ_k + (1 - m) θ_q
        """
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                     self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """
        keys: [B, dim] tensor of new keys
        """
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        # Ensure K % batch_size == 0 for simplicity
        assert self.K % batch_size == 0  

        # Replace the oldest keys
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        """
        im_q: query images (view1) [B, 3, H, W]
        im_k: key   images (view2) [B, 3, H, W]
        Returns: logits [B, 1+K], labels [B] (all zeros)
        """
        # 1) Compute query features
        _, q = self.encoder_q(im_q)        # [B, dim]
        q = nn.functional.normalize(q, dim=1)

        # 2) Compute key features with no_grad
        with torch.no_grad():
            self._momentum_update_key_encoder()
            _, k = self.encoder_k(im_k)    # [B, dim]
            k = nn.functional.normalize(k, dim=1)

        # 3) Compute logits
        # Positive logits: [B, 1]
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # Negative logits: [B, K]
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1)  # [B, 1+K]
        logits /= self.T

        # Labels: positive at index 0
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

        # 4) Enqueue and dequeue
        self._dequeue_and_enqueue(k)

        return logits, labels

################################################################################
# 3) BYOL Wrapper
################################################################################
class BYOL(nn.Module):
    """
    BYOL wrapper:
     - online_encoder: backbone + projection head
     - predictor: small MLP
     - target_encoder: momentum copy of online
    """
    def __init__(self, base_encoder, out_dim=128, hidden_dim=512, m=0.996):
        super().__init__()
        self.online_encoder = base_encoder(out_dim=out_dim)
        self.predictor = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

        # Create target encoder as a copy of online (no gradients)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        self.m = m  # momentum coefficient

    @torch.no_grad()
    def _momentum_update_target(self):
        """
        θ_target ← m θ_target + (1-m) θ_online
        """
        for param_o, param_t in zip(self.online_encoder.parameters(),
                                    self.target_encoder.parameters()):
            param_t.data = param_t.data * self.m + param_o.data * (1.0 - self.m)

    def forward(self, x1, x2):
        """
        x1, x2: two augmented views [B, 3, H, W]
        Returns BYOL loss (MSE between p and stop_grad(z_t))
        """
        # 1) Online pass
        h1, z1 = self.online_encoder(x1)  # z1: [B, out_dim]
        h2, z2 = self.online_encoder(x2)

        p1 = self.predictor(z1)  # [B, out_dim]
        p2 = self.predictor(z2)

        # 2) Target pass (no grad)
        with torch.no_grad():
            _, z1_t = self.target_encoder(x1)  # [B, out_dim]
            _, z2_t = self.target_encoder(x2)

        # 3) Normalize
        p1 = nn.functional.normalize(p1, dim=1)
        p2 = nn.functional.normalize(p2, dim=1)
        z1_t = nn.functional.normalize(z1_t, dim=1)
        z2_t = nn.functional.normalize(z2_t, dim=1)

        # 4) Compute BYOL loss: 2 - 2 * cos_sim
        loss1 = 2 - 2 * (p1 * z2_t).sum(dim=1)
        loss2 = 2 - 2 * (p2 * z1_t).sum(dim=1)
        loss = (loss1 + loss2).mean()
        return loss

    def update_target(self):
        self._momentum_update_target()

################################################################################
# 4) DINO Skeleton (ResNet-18 + prototypes head)
################################################################################
class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim=1024, hidden_dim=512):
        """
        in_dim: backbone output dim (512 for ResNet-18)
        out_dim: number of prototypes (e.g., 1024)
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        # Temperature parameters
        self.student_temp = 0.1
        self.teacher_temp = 0.04
        # Center for teacher
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward_student(self, x):
        logits = self.mlp(x)            # [B, out_dim]
        logits = logits / self.student_temp
        return logits

    @torch.no_grad()
    def forward_teacher(self, x):
        logits = self.mlp(x)            # [B, out_dim]
        logits = (logits - self.center) / self.teacher_temp
        return torch.softmax(logits, dim=-1)

    @torch.no_grad()
    def update_center(self, teacher_outputs):
        """
        teacher_outputs: [B, out_dim], after softmax
        """
        batch_center = torch.mean(teacher_outputs, dim=0, keepdim=True)
        decay = 0.9
        self.center = self.center * decay + batch_center * (1 - decay)

class DINO(nn.Module):
    """
    DINO wrapper:
     - student_backbone (ResNet-18) + student_head (DINOHead)
     - teacher_backbone + teacher_head (momentum copy)
     - Multi-crop loss
    """
    def __init__(self,
                 backbone_fn,
                 in_dim=512,
                 num_prototypes=1024,
                 m=0.996):
        super().__init__()
        # Student
        self.student_backbone = backbone_fn()
        self.student_head = DINOHead(in_dim=in_dim, out_dim=num_prototypes)

        # Teacher
        self.teacher_backbone = copy.deepcopy(self.student_backbone)
        self.teacher_head     = copy.deepcopy(self.student_head)
        for p in self.teacher_backbone.parameters():
            p.requires_grad = False
        for p in self.teacher_head.parameters():
            p.requires_grad = False

        self.m = m  # momentum for teacher

    @torch.no_grad()
    def _momentum_update_teacher(self):
        # Update backbone
        for param_s, param_t in zip(self.student_backbone.parameters(),
                                    self.teacher_backbone.parameters()):
            param_t.data = param_t.data * self.m + param_s.data * (1.0 - self.m)
        # Update head
        for param_s, param_t in zip(self.student_head.parameters(),
                                    self.teacher_head.parameters()):
            param_t.data = param_t.data * self.m + param_s.data * (1.0 - self.m)

    def forward(self, crops):
        """
        crops: list of augmented images
               e.g. [global1, global2, local1, local2, local3, local4]
        Returns: DINO loss
        """
        B = crops[0].size(0)
        # 1) Student: forward on all crops
        student_logits = []
        for crop in crops:
            feat = self.student_backbone(crop)           # [B, in_dim]
            logit = self.student_head.forward_student(feat)  # [B, num_prototypes]
            student_logits.append(logit)

        # 2) Teacher: forward on only the first two global crops
        with torch.no_grad():
            teacher_probs = []
            for crop in crops[:2]:
                feat_t = self.teacher_backbone(crop)
                prob  = self.teacher_head.forward_teacher(feat_t)  # [B, num_prototypes]
                teacher_probs.append(prob)
            # Concatenate to shape [2B, num_prototypes]
            teacher_probs = torch.cat(teacher_probs, dim=0)

        # 3) Cross-entropy between teacher_probs and student_logits for each crop
        total_loss = 0.0
        n_crops = len(crops)
        for v in range(n_crops):
            logits_v = student_logits[v]                  # [B, P]
            log_probs = torch.log_softmax(logits_v, dim=1)  # [B, P]
            # Repeat to match [2B, P]
            log_probs = torch.cat([log_probs, log_probs], dim=0)
            loss_v = - (teacher_probs * log_probs).sum(dim=1).mean()
            total_loss += loss_v
        total_loss /= n_crops

        return total_loss

    def update_teacher(self):
        self._momentum_update_teacher()
        # After computing teacher outputs, call `self.student_head.update_center(...)` in training code
