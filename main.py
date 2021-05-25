from tools import *
from paint import *
from decoder import *

assert __name__ == '__main__', 'Main program startup error.'


def main():
    decoder = placeholder  # optional
    tensor2img = transforms.ToPILImage()

    generator = GenerativeAlgorithm()
    classifier = Ensemble(*[f'classifier-{index}.pth' for index in range(1, 6)])

    def generate(tensor: torch.Tensor):
        image = tensor2img(tensor)
        index = torch.argmax(classifier.forward(torch.unsqueeze(classifier.preprocess(image), 0).to(DEVICE)))
        generator.get_image(decoder(index.data)).resize((700, 700), resample=Image.NEAREST).show()

    app = Application(generate)
    app.start()


main()
