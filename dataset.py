import numpy as np

class Dataset():
    def __init__(self, fn_labels, fn_images):
        self.fn_labels = fn_labels
        self.fn_images = fn_images

        self.images = []
        self.img_count = 0
        self.img_size = [0, 0]

        self.labels = []
    
    def load(self, count):
        self.img_count = count
        with open(self.fn_images, 'rb') as file:
            magic_nb = int.from_bytes(file.read(4), "big")
            total_img = int.from_bytes(file.read(4), "big")
            self.img_size[1] = int.from_bytes(file.read(4), "big")
            self.img_size[0] = int.from_bytes(file.read(4), "big")
            
            print(f'mn: {magic_nb}, total_img:{total_img}, nb_images:{self.img_count}, size: {self.img_size[0]}x{self.img_size[1]}')
            data = 0
            for i in range(count):
                image = np.zeros((self.img_size[0], self.img_size[1]), dtype=np.float32)
                for r in range(self.img_size[1]):
                    for c in range(self.img_size[0]):
                        data = int.from_bytes(file.read(1), "big")
                        image[r,c] = float(data / 255.0)
                self.images.append(image)
        
        
        with open(self.fn_labels, 'rb') as file:
            magic_nb = int.from_bytes(file.read(4), "big")
            total_labels = int.from_bytes(file.read(4), "big")
            for i in range(count):
                self.labels.append(int.from_bytes(file.read(1), "big"))
        
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)