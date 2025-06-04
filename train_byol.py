import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets

from models import BYOL, ResNetEncoder
from transforms import ContrastiveTransform

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.996)
    parser.add_argument("--save_path", type=str, default="byol.pth")
    return parser.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) DataLoader
    train_transform = ContrastiveTransform(base_size=32)
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

    # 2) Model, optimizer
    byol_model = BYOL(
        base_encoder=lambda out_dim: ResNetEncoder(base="resnet18", out_dim=out_dim),
        out_dim=128,
        hidden_dim=512,
        m=args.momentum
    ).to(device)
    optimizer = optim.Adam(byol_model.parameters(), lr=args.lr, weight_decay=1e-4)

    # 3) Training loop
    for epoch in range(args.epochs):
        byol_model.train()
        total_loss = 0.0
        for (x1, x2), _ in train_loader:
            x1 = x1.to(device)
            x2 = x2.to(device)

            loss = byol_model(x1, x2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            byol_model.update_target()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[BYOL] Epoch [{epoch+1}/{args.epochs}]  Loss: {avg_loss:.4f}")

    # 4) Save online encoder only (backbone + projection head)
    torch.save(byol_model.online_encoder.state_dict(), args.save_path)
    print(f"BYOL model saved to {args.save_path}")

if __name__ == "__main__":
    main()
