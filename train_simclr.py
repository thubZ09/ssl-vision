import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets

from models import ResNetEncoder
from transforms import ContrastiveTransform
from utils import NTXentLoss

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--save_path", type=str, default="simclr.pth")
    return parser.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # dataLoader
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

    #model, optimizer, loss
    model = ResNetEncoder(base="resnet18", out_dim=128).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = NTXentLoss(batch_size=args.batch_size, temperature=args.temperature)

    # train loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for (x_i, x_j), _ in train_loader:
            x_i = x_i.to(device)
            x_j = x_j.to(device)

            _, z_i = model(x_i)  # [B, 128]
            _, z_j = model(x_j)  # [B, 128]
            z = torch.cat([z_i, z_j], dim=0)  # [2B, 128]

            loss = criterion(z)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[SimCLR] Epoch [{epoch+1}/{args.epochs}]  Loss: {avg_loss:.4f}")

    # Save 
    torch.save(model.state_dict(), args.save_path)
    print(f"SimCLR model saved to {args.save_path}")

if __name__ == "__main__":
    main()
