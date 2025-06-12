import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

from models import ResNetEncoder

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True,
                        choices=["simclr", "moco", "byol", "dino"])
    parser.add_argument("--pretrained_path", type=str, required=True,
                        help="Path to SSL checkpoint")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=30.0)
    return parser.parse_args()

def load_backbone(args, device):
   
    backbone = ResNetEncoder(base="resnet18", out_dim=128)
    if args.method == "simclr":
        state = torch.load(args.pretrained_path, map_location=device)
        backbone.load_state_dict(state)
        repr_model = backbone.backbone  # freeze this

    elif args.method == "moco":
        state = torch.load(args.pretrained_path, map_location=device)
        backbone.encoder_q.backbone.load_state_dict(state["encoder_q"])
        repr_model = backbone.backbone

    elif args.method == "byol":
        state = torch.load(args.pretrained_path, map_location=device)
        backbone.load_state_dict(state)
        repr_model = backbone.backbone

    elif args.method == "dino":
        state = torch.load(args.pretrained_path, map_location=device)
        repr_model = backbone.backbone
        repr_model.load_state_dict(state["backbone"])

    else:
        raise ValueError("Unknown method")

    for param in repr_model.parameters():
        param.requires_grad = False
    repr_model.eval()
    return repr_model

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #load pretrained backbone
    repr_model = load_backbone(args, device).to(device)

    #create linear classifier
    # resnet-18 backbone output dim = 512
    linear_classifier = nn.Linear(512, 10).to(device)

    #dataLoader for CIFAR-10
    transform_train = T.Compose([
        T.RandomResizedCrop(32),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465),
                    (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465),
                    (0.2470, 0.2435, 0.2616)),
    ])
    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=False, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=False, transform=transform_test
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=2)

    # optimizer + loss
    optimizer = optim.SGD(linear_classifier.parameters(),
                          lr=args.lr, momentum=0.9, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()

    # training loop
    for epoch in range(args.epochs):
        linear_classifier.train()
        total_loss, correct, total = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                h = repr_model(x)    # [B, 512]
            logits = linear_classifier(h)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            _, preds = logits.max(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

        train_acc = correct / total * 100
        train_loss = total_loss / total

        # evaluate
        linear_classifier.eval()
        correct_test, total_test = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                h = repr_model(x)
                logits = linear_classifier(h)
                _, preds = logits.max(dim=1)
                correct_test += (preds == y).sum().item()
                total_test += x.size(0)
        test_acc = correct_test / total_test * 100

        print(f"[Eval] Epoch [{epoch+1}/{args.epochs}]  "
              f"Train Loss: {train_loss:.4f}  "
              f"Train Acc: {train_acc:.2f}%  "
              f"Test Acc: {test_acc:.2f}%")

    # save the linear classifier
    torch.save(linear_classifier.state_dict(), f"linear_{args.method}.pth")
    print(f"Linear head saved to linear_{args.method}.pth")

if __name__ == "__main__":
    main()
