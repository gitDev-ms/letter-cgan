from nets import *
import matplotlib.pyplot as plt

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn.functional as func

from string import ascii_uppercase, ascii_lowercase
from PIL import Image, ImageFilter
from io import BytesIO
import time
import os

assert __name__ != '__main__', 'Module startup error.'

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
LETTERS = ascii_uppercase + ascii_lowercase
NORMALIZATION = {'mean': 0.1322, 'std': 0.2827}

DATA_ROOT = r'D:\App\Datasets\EMNIST\dataset'
DIRECTORY = 'models'
LOGS = 'logs'


def log_save(samples: torch.Tensor):
    tensor2img = transforms.ToPILImage()
    figure = plt.figure(figsize=(26, 2), dpi=224, frameon=False)
    plt.subplots_adjust(wspace=0, hspace=0)

    for index, sample in enumerate(samples):
        figure.add_subplot(2, 26, index + 1)
        plt.imshow(tensor2img(sample).convert('RGB'))
        plt.axis('off')

    files = os.listdir(LOGS)
    file = f'samples-{int(files[-1][8:13]) + 1:0>5}.png' if files else 'samples-00000.png'
    plt.savefig(os.path.join(LOGS, file))

    plt.close()


class DataSet(ImageFolder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.class_to_idx = {index: int(class_) for class_, index in self.class_to_idx.items()}
        self.classes = list(self.class_to_idx.keys())

    def __getitem__(self, index: int) -> tuple:
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample.to(DEVICE), torch.tensor(self.class_to_idx[target], device=DEVICE)

    @property
    def std(self) -> torch.Tensor:
        return torch.mean(torch.stack([torch.std(sample[0]) for sample in self]))

    @property
    def mean(self) -> torch.Tensor:
        return torch.mean(torch.stack([torch.mean(sample[0]) for sample in self]))


class LearningManager:
    def __init__(self, path2g: str = None, path2d: str = None):
        self.normalize = transforms.Normalize(*NORMALIZATION.values())
        preprocess = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            self.normalize
        ])
        self.dataset = DataSet(os.path.join(DATA_ROOT, 'train'), transform=preprocess)

        self.generator = Generator() if path2g is None else torch.load(os.path.join(DIRECTORY, 'CGAN', path2g))
        self.discriminator = Discriminator() if path2d is None else torch.load(os.path.join(DIRECTORY, 'CGAN', path2d))
        self.generator.to(DEVICE)
        self.discriminator.to(DEVICE)

        if path2g is None:
            self.generator.apply(weights_init)
        if path2d is None:
            self.discriminator.apply(weights_init)

        self.fixed_noise_seed = 1415926535

    def save_models(self, path2g: str, path2d: str):
        self.generator.eval()
        self.discriminator.eval()

        torch.save(self.generator, os.path.join(DIRECTORY, 'CGAN', path2g))
        torch.save(self.discriminator, os.path.join(DIRECTORY, 'CGAN', path2d))

    def train(self, epochs: int, seed: int = None):
        if seed is None:
            seed = torch.randint(-2 ** 32, 2 ** 32, (1,))

        torch.random.manual_seed(self.fixed_noise_seed)
        fixed_noise = (torch.randn((52, 128), device=DEVICE),
                       func.one_hot(torch.arange(52, device=DEVICE), num_classes=52))

        torch.random.manual_seed(seed)
        loader = DataLoader(self.dataset, batch_size=128, shuffle=True)

        self.generator.train()
        self.discriminator.train()

        criterion = nn.BCELoss()
        optimizer4g = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        optimizer4d = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

        start = time.time()
        for epoch in range(epochs):
            g_loss, d_loss = [], []

            for data in loader:
                samples, labels = data
                mb_size = samples.size(0)
                labels = func.one_hot(labels, num_classes=52)
                noise = torch.randn((mb_size, 128), device=DEVICE)

                optimizer4d.zero_grad()
                self.discriminator.zero_grad()

                targets = torch.ones(mb_size, device=DEVICE)
                prediction = self.discriminator(samples, labels).view(-1)

                cost4d_real = criterion(prediction, targets)
                cost4d_real.backward()

                fake_samples = self.normalize(self.generator(noise, labels))
                targets = torch.zeros(mb_size, device=DEVICE)
                prediction = self.discriminator(fake_samples.detach(), labels).view(-1)

                cost4d_fake = criterion(prediction, targets)
                cost4d_fake.backward()

                optimizer4d.step()
                d_loss.append((cost4d_real + cost4d_fake).item())

                optimizer4g.zero_grad()
                self.generator.zero_grad()

                targets = torch.ones(mb_size, device=DEVICE)
                prediction = self.discriminator(fake_samples, labels).view(-1)

                cost4g = criterion(prediction, targets)
                cost4g.backward()

                optimizer4g.step()
                g_loss.append(cost4g.item())

            with torch.no_grad():
                log_save(self.generator(*fixed_noise).detach().to('cpu'))
            self.generator.zero_grad()

            print(f'Epoch number: {epoch + 1}',
                  f'Mean generator loss: {torch.mean(torch.tensor(g_loss)).item()}',
                  f'Mean discriminator loss: {torch.mean(torch.tensor(d_loss)).item()}',
                  sep='\n', end='\n\n')

        print('Total time: ', time.time() - start, end='\n\n')


class GenerativeAlgorithm:
    def __init__(self, generator: Generator = None):
        if generator is None:
            generator = torch.load(os.path.join(DIRECTORY, 'CGAN', 'generator.pth'))
        self.generator = generator
        self.generator.to(DEVICE)
        self.generator.eval()

        self.discriminator = torch.load(os.path.join(DIRECTORY, 'CGAN', 'discriminator.pth'))
        self.discriminator.to(DEVICE)
        self.discriminator.eval()

        self.tensor2img = transforms.ToPILImage()
        self.tensor_transforms = transforms.Compose([
            transforms.CenterCrop(24),
            transforms.Pad(2)
        ])

        self.img_transforms = (
            ImageFilter.GaussianBlur(radius=1),
            ImageFilter.SHARPEN,
        )

        self.get_sample(0, depth=1000)  # computer preparation

    def get_sample(self, label: int, depth: int = 20) -> torch.Tensor:
        x = torch.randn((depth, 128), device=DEVICE)
        y = func.one_hot(torch.tensor([label] * depth, device=DEVICE), num_classes=52)
        with torch.no_grad():
            tensor = self.tensor_transforms(self.generator(x, y))
            return tensor[torch.argmax(self.discriminator(tensor, y))]

    def get_image(self, label: int, depth: int = 20) -> Image.Image:
        tensor = self.get_sample(label, depth)
        image = self.tensor2img(tensor).convert('RGB')
        for transform in self.img_transforms:
            image = image.filter(transform)

        return image

    @property
    def samples(self) -> Image.Image:
        figure = plt.figure(figsize=(26, 2), dpi=224, frameon=False)
        plt.subplots_adjust(wspace=0, hspace=0)

        for label in range(52):
            figure.add_subplot(2, 26, label + 1)
            plt.imshow(self.get_image(label))
            plt.axis('off')

        buf = BytesIO()
        plt.savefig(buf)
        buf.seek(0)

        plt.close()
        return Image.open(buf)


class ClassifierManager:
    def __init__(self, path: str = None):
        train_preprocess = transforms.Compose([
            transforms.Grayscale(),
            transforms.RandomAffine(degrees=(0, 0), translate=(0, 0)),
            transforms.ToTensor(),
            transforms.Normalize(*NORMALIZATION.values())
        ])
        test_preprocess = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(*NORMALIZATION.values())
        ])

        self.train_ds = DataSet(os.path.join(DATA_ROOT, 'train'), transform=train_preprocess)
        self.test_ds = DataSet(os.path.join(DATA_ROOT, 'test'), transform=test_preprocess)

        self.model = Classifier() if path is None else torch.load(os.path.join(DIRECTORY, 'classifier', path))
        self.model.to(DEVICE)

    def save(self, path: str):
        self.model.eval()
        torch.save(self.model, os.path.join(DIRECTORY, 'classifier', path))

    def train(self, epochs: int, seed: int = None):
        if seed is not None:
            torch.random.manual_seed(seed)

        loader = DataLoader(self.train_ds, batch_size=128, shuffle=True)
        self.model.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0002)

        start = time.time()
        for epoch in range(epochs):
            accuracy, loss = [], []

            for data in loader:
                optimizer.zero_grad()

                samples, targets = data
                prediction = self.model(samples)
                cost = criterion(prediction, targets)

                cost.backward()
                optimizer.step()

                accuracy.append(torch.sum(torch.argmax(prediction, dim=1) == targets) / targets.size(0))
                loss.append(cost.data)

            print(f'Epoch number: {epoch + 1}',
                  f'Mean accuracy: {torch.mean(torch.tensor(accuracy)).item()}',
                  f'Mean loss: {torch.mean(torch.tensor(loss)).item()}',
                  sep='\n', end='\n\n')
        print(f'Total time: {time.time() - start}', end='\n\n')

    def test(self):
        loader = DataLoader(self.test_ds, batch_size=self.test_ds.__len__() // 52)
        self.model.eval()

        accuracy = []
        start = time.time()

        for data in loader:
            samples, targets = data

            with torch.no_grad():
                prediction = self.model(samples)
            accuracy.append(torch.sum(torch.argmax(prediction, dim=1) == targets))

        print(f'Accuracy: {sum(accuracy) / self.test_ds.__len__()}',
              f'Total time: {time.time() - start}',
              sep='\n', end='\n\n')


class Ensemble:
    def __init__(self, *models: str):
        self.nets = [torch.load(os.path.join(DIRECTORY, 'classifier', path)).to(DEVICE) for path in models]
        self.preprocess = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(*NORMALIZATION.values())
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([model(x) for model in self.nets]).sum(dim=0)

    def test(self):
        dataset = DataSet(os.path.join(DATA_ROOT, 'test'), transform=self.preprocess)
        loader = DataLoader(dataset, batch_size=dataset.__len__() // 52)

        accuracy = []
        start = time.time()

        for data in loader:
            samples, targets = data

            with torch.no_grad():
                prediction = self.forward(samples)
            accuracy.append(torch.sum(torch.argmax(prediction, dim=1) == targets))

        print(f'Accuracy: {sum(accuracy) / dataset.__len__()}',
              f'Total time: {time.time() - start}',
              sep='\n', end='\n\n')
