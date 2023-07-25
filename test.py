import pathlib
import argparse

import torch
from torch.utils.data import DataLoader
from PIL import Image

from cp_net import SkinNet
from train import DATA_TRANSFORMS, DATASETS

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="path to the file containing the trained model.")
    parser.add_argument("--top_k", default=5, type=int, help="top predictions to return")
    parser.add_argument("--image_path", type=str, help="path to image for inference.")
    parser.add_argument("--batch_size", default=4, type=int, help="number of data samples to load at once.")

    return parser.parse_args()

def test(model_path, image_path, top_k, batch_size, num_class=2, verbose=True):
    DATALOADERS = {
            "train" : DataLoader(dataset=DATASETS["train"], batch_size=batch_size, shuffle=True),
            #"test" : DataLoader(dataset=DATASETS["test"], batch_size=batch_size, shuffle=True),
            "validation" : DataLoader(dataset=DATASETS["validation"], batch_size=batch_size, shuffle=True)
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SkinNet(num_class, pretrained=False)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Couldn't load pretrained model from: {model_path}")

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        if isinstance(image_path, str):
            images = [image_path] if pathlib.ntpath.isfile(image_path) else \
                    list(map(
                            lambda x : pathlib.ntpath.join(image_path, x),
                            pathlib.ntpath.os.listdir(image_path)
                        ))
            for img_path in images:
                print("-"*50, img_path, "-"*50, sep='\n')
                img = Image.open(img_path).convert("RGB")
                img = DATA_TRANSFORMS["test"](img).unsqueeze(0).to(device)
                output = model(img)
                output_probs = torch.nn.functional.softmax(output, dim=1)
                idx_to_class = {V:K for K, V in DATASETS["train"].class_to_idx.items()}
                probs, classes = torch.topk(output_probs, top_k)
                classes = [idx_to_class[i] for i in classes.squeeze().tolist()]
                probs = probs.squeeze().tolist()
                if verbose:
                    print("\n".join([f"{c:<15}--->{p:>10.2f}" for c,p in zip(classes, probs)]), end="\n\n")
                return probs, classes
        else:
            running_accuracy = 0.0
            for imgs, labels in DATALOADERS["test"]:
                imgs = imgs.to(device)
                labels = labels.to(device)

                outputs = model(imgs)
                _, predictions = torch.max(outputs, dim=1)
                running_accuracy += torch.sum(predictions==labels.data)
            if verbose:
                print(f"Test accuracy: {(running_accuracy.double() / len(DATASETS['test']))*100 : 0.2f}%")
            return None


if __name__ == "__main__":
    args = arg_parser()
    test(args.model_path, args.image_path, args.top_k, args.batch_size)
    torch.cuda.empty_cache()

