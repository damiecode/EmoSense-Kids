import pathlib
import argparse

import torch
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from cp_util import DATA_TRANSFORMS
from cp_net import CPNet

torch.manual_seed(42)
DATA_DIR = pathlib.Path(__file__).parent/'dataset'

DATASETS = {
            "train" : ImageFolder(root=DATA_DIR/"train", transform=DATA_TRANSFORMS['train']),
            #"test" : ImageFolder(root=DATA_DIR/"test", transform=DATA_TRANSFORMS['test']),
            "validation" : ImageFolder(root=DATA_DIR/"validation", transform=DATA_TRANSFORMS['test'])
        }

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=50, type=int, help="number of training iterations.")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="learning rate for parameter update.")
    parser.add_argument("--batch_size", default=4, type=int, help="number of data samples to load at once.")
    parser.add_argument("attempt", type=int, help="current training attempt")
    return parser.parse_args()

def train(attempt, epochs, learning_rate, batch_size):
    DATALOADERS = {
            "train" : DataLoader(dataset=DATASETS["train"], batch_size=batch_size, shuffle=True),
            #"test" : DataLoader(dataset=DATASETS["test"], batch_size=batch_size, shuffle=True),
            "validation" : DataLoader(dataset=DATASETS["validation"], batch_size=batch_size, shuffle=True)
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CPNet(len(DATASETS["train"].classes)).to(device)
    optimizer = optim.SGD(
                model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True,
                weight_decay=1e-2
            )
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    criterion = torch.nn.CrossEntropyLoss()

    best_accuracy = 0.0
    loss_history = {"train":[], "validation":[]}
    accuracy_history = {"train":[], "validation":[]}

    attempt_path = DATA_DIR.parent/"models"/f"{attempt}"
    if not pathlib.Path.exists(attempt_path):
        pathlib.os.makedirs(attempt_path)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}", "-"*10, sep="\n")
        for phase in ["train", "validation"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_accuracy = 0.0

            for imgs, labels in DATALOADERS[phase]:
                imgs = imgs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                
                with torch.set_grad_enabled((phase=="train")):
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                    
                running_loss += loss.item() * imgs.size(0)
                running_accuracy += torch.sum(preds==labels.data)
                
            epoch_loss = running_loss / len(DATASETS[phase])
            epoch_accuracy = running_accuracy.double() / len(DATASETS[phase])
            loss_history[phase].append(epoch_loss)
            accuracy_history[phase].append(epoch_accuracy)

            print(f"{phase} Loss: {epoch_loss:.5f}, Accuracy: {epoch_accuracy:.5f}")
            
            if phase == "validation" and epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                torch.save(
                            model.state_dict(),
                            attempt_path/f"attempt-{attempt}-best.pth"
                        )
            print()

        scheduler.step(loss_history["validation"][-1])
    torch.save(
                {"loss_history" : loss_history, "accuracy_history" : accuracy_history},
                attempt_path/f"attempt-{attempt}-history.pth"
            )

if __name__ == "__main__":
    args = arg_parser()
    train(args.attempt, args.epochs, args.learning_rate, args.batch_size)
    torch.cuda.empty_cache()
