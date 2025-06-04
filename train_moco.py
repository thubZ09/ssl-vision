import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets

from models import MoCo, ResNetEncoder
from transforms import ContrastiveTransform

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--queue_size", type=int, default=4096)
    parser.add_argument("--momentum", type=float, default=0.99)
    parser.add_argument("--save_path", type=str, default="moco.pth")
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

    # 2) Model, optimizer, loss
    moco_model = MoCo(
        base_encoder=lambda out_dim: ResNetEncoder(base="resnet18", out_dim=out_dim),
        dim=128,
        K=args.queue_size,
        m=args.momentum,
        T=0.2
    ).to(device)
    optimizer = optim.SGD(moco_model.encoder_q.parameters(),
                          lr=args.lr, momentum=0.9, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # 3) Training loop
    for epoch in range(args.epochs):
        moco_model.train()
        total_loss = 0.0
        for (im_q, im_k), _ in train_loader:
            im_q = im_q.to(device)
            im_k = im_k.to(device)

            logits, labels = moco_model(im_q, im_k)  # [B, 1+K], [B]
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[MoCo] Epoch [{epoch+1}/{args.epochs}]  Loss: {avg_loss:.4f}")

    # 4) Save only the query encoder (encoder_q + its projection head)
    state = {
        "encoder_q": moco_model.encoder_q.state_dict(),
    }
    torch.save(state, args.save_path)
    print(f"MoCo model saved to {args.save_path}")

if __name__ == "__main__":
    main()
