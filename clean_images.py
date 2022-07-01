from PIL import Image
import os

class CleanImages():
    def __init__(self, path):
        self.path = path
        self.dirs = os.listdir(self.path)
        self.final_size = 512

    def clean_and_save(self, **kwargs):
        if 'amount' in kwargs:
            amount = kwargs['amount']
        else:
            amount = len(self.dirs)
        for file in self.dirs[:amount]:
            image = Image.open('images/' + file)
            resized_image = self.resize_image(self.final_size, image)
            if not os.path.exists('resized_images'):
                os.makedirs('resized_images')
            resized_image.save(f'resized_images/resized_{file}')

    def resize_image(self, final_size, image):
        image_size = image.size
        max_dimensions = max(image.size)
        size_ratio = float(final_size) / float(max_dimensions)
        new_image_size = tuple([int(x*size_ratio) for x in image_size])
        image = image.resize(new_image_size, Image.ANTIALIAS)
        new_image = Image.new("RGB", (final_size, final_size))
        new_image.paste(image, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
        return new_image

if __name__ == '__main__':
    clean = CleanImages('images/')
    clean.clean_and_save()