import torchvision.transforms as T
# contrastiveTransform (SimCLR, MoCo, BYOL)

class ContrastiveTransform:
    def __init__(self, base_size=32):
        s = 1.0
        self.transform = T.Compose([
            T.RandomResizedCrop(size=base_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(
                [T.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)],
                p=0.8
            ),
            T.RandomGrayscale(p=0.2),
            T.RandomApply(
                [T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))],
                p=0.5
            ),
            T.ToTensor(),
            T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                        std=(0.2470, 0.2435, 0.2616)),
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)

################################################################################
# dino Multi-Crop Transform

class DINOTransform:
    def __init__(self,
                 global_crop_size=32,
                 local_crop_size=16,
                 local_crops_number=4):
        """
        For CIFAR-10 we shrink the sizes: 
        Two global crops at 32×32, four local crops at 16×16
        """
        self.global_transforms = T.Compose([
            T.RandomResizedCrop(global_crop_size, scale=(0.4, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            T.ToTensor(),
            T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                        std=(0.2470, 0.2435, 0.2616)),
        ])
        self.local_transforms = T.Compose([
            T.RandomResizedCrop(local_crop_size, scale=(0.05, 0.4)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            T.ToTensor(),
            T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                        std=(0.2470, 0.2435, 0.2616)),
        ])
        self.local_crops_number = local_crops_number

    def __call__(self, img):
        crops = []
        # two global crops
        for _ in range(2):
            crops.append(self.global_transforms(img))
        # four local crops
        for _ in range(self.local_crops_number):
            crops.append(self.local_transforms(img))
        return crops
