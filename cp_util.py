import random

from torchvision import transforms

class RandomRotationTransform:

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, X):
        angle = random.choice(self.angles)
        return transforms.functional.rotate(X, angle)


DATA_TRANSFORMS = {
            "train" : transforms.Compose([
                    transforms.Resize(256),
                    RandomRotationTransform([90, 180, 270]),
                    # transforms.RandomResizedCrop(300),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    #transforms.Normalize(
                    #        mean=[0.485, 0.456, 0.406],
                    #        std=[0.229, 0.224, 0.225]
                    #    )
                ]),
            "test" : transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    #transforms.Normalize(
                    #        mean=[0.485, 0.456, 0.406],
                    #        std=[0.229, 0.224, 0.225]
                    #    )
                ])
        }

