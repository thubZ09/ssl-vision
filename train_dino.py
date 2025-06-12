import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets

from models import DINO, ResNetEncoder
from transforms import DINOTransform

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--momentum", type=float, default=0.996)
    parser.add_argument("--num_prototypes", type=int, default=1024)
    parser.add_argument("--save_path", type=str, default="dino.pth")
    return parser.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # dataLoader
    train_transform = DINOTransform(
        global_crop_size=32,
        local_crop_size=16,
        local_crops_number=4
    )
    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=lambda x: train_transform(x)
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True
    )

    # model, optimizer
    dino_model = DINO(
        backbone_fn=lambda: ResNetEncoder(base="resnet18", out_dim=128).backbone,
        in_dim=512,
        num_prototypes=args.num_prototypes,
        m=args.momentum
    ).to(device)
    optimizer = optim.SGD(
        list(dino_model.student_backbone.parameters()) +
        list(dino_model.student_head.parameters()),
        lr=args.lr, momentum=0.9, weight_decay=1e-4
    )

    # training loop
    for epoch in range(args.epochs):
        dino_model.train()
        total_loss = 0.0
        for crops, _ in train_loader:
            # crops: list of [B, 3, H, W], length = 2 global + 4 local = 6
            crops = [c.to(device) for c in crops]
            loss = dino_model(crops)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update teacher and center
            with torch.no_grad():
                # 1) compute teacher outputs on global crops
                teacher_feats = []
                for gx in crops[:2]:
                    feat_t = dino_model.teacher_backbone(gx)
                    prob  = dino_model.teacher_head.forward_teacher(feat_t)
                    teacher_feats.append(prob)
                teacher_probs = torch.cat(teacher_feats, dim=0)  # [2B, P]
                dino_model.student_head.update_center(teacher_probs)
                # 2) momentum update teacher
                dino_model.update_teacher()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[DINO] Epoch [{epoch+1}/{args.epochs}]  Loss: {avg_loss:.4f}")

    # sSave only student backbone + head (for downstream)
    state = {
        "backbone": dino_model.student_backbone.state_dict(),
        "head":     dino_model.student_head.state_dict(),
    }
    torch.save(state, args.save_path)
    print(f"DINO model saved to {args.save_path}")

if __name__ == "__main__":
    main()
