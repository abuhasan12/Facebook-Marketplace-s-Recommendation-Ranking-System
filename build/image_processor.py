from torchvision import transforms

class ImageProcessor:
    def __init__(self):
        self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4217, 0.3923, 0.3633],
                                        std=[0.3117, 0.2967, 0.2931])
            ])

        self.transform_Gray = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Lambda(self.repeat_channel),
            transforms.Normalize(mean=[0.4217, 0.3923, 0.3633],
                                    std=[0.3117, 0.2967, 0.2931])
        ])

    @staticmethod
    def repeat_channel(x):
            return x.repeat(3, 1, 1)

    def __call__(self, image):
        if image.mode != 'RGB':
            image = self.transform_Gray(image)
        else:
            image = self.transform(image)
        
        image = image[None, :, :, :]
        return image