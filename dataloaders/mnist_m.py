import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MnistM(Dataset):
    def __init__(self, split='train'):
        self.split = split

        if split == 'train':
            self.img_path = '/home/ubuntu/nadav/data/mnist_m/mnist_m_train'
            self.label_file_path = '/home/ubuntu/nadav/data/mnist_m/mnist_m_train_labels.txt'
        else:  # split == 'test'
            self.img_path = '/home/ubuntu/nadav/data/mnist_m/mnist_m_test'
            self.label_file_path = '/home/ubuntu/nadav/data/mnist_m/mnist_m_test_labels.txt'

        self.image_paths = []
        self.file2labels = self._read_labels()
        # self._load_images()

    def __len__(self):
        return len(self.file2labels)

    def __getitem__(self, index):
        path = self.image_paths[index]
        image = Image.open(path)
        label = self.file2labels[path]
        composed_transforms = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.4581593085386675, 0.46234912491916985, 0.40847689820987465),
                                 (0.23861309585320542, 0.22387564111405261, 0.24443159305397594))
        ])
        return composed_transforms(image), label

    def _read_labels(self):
        f = open(self.label_file_path, mode='r')
        lines = f.readlines()
        file2labels = {}
        for line in lines:
            name = line[:12]
            path = os.path.join(self.img_path, name)
            label = int(line[13])
            file2labels[path] = label
            self.image_paths.append(path)
        return file2labels

    # def _load_images(self):
    #     assert len(self.file2labels) != 0, 'labels should be loaded prior to images'
    #     for f_name in self.file2labels.keys():
    #         image = Image.open(os.path.join(self.img_path, f_name))
    #         self.images.append(image)
    #         self.labels.append(self.file2labels[f_name])


