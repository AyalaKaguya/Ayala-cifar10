import torch
from PIL import Image
from ModelAyala import Ayala
from torchvision import transforms, datasets

dataset = datasets.CIFAR10(
    root="./Dataset",
    train=True,
)

ayala = Ayala()
ayala.load_state_dict(torch.load("./ayala-module.pth"))
image = Image.open("./dog.png")
image = image.convert("RGB")
transf = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])
image = transf(image)
image = torch.reshape(image, (1, 3, 32, 32))
ayala.eval()
with torch.no_grad():
    output = ayala(image)
    target = int(output.argmax(1))
    print(dataset.classes[target])
