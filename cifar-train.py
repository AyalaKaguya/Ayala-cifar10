import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch import optim

from ModelAyala import Ayala


learning_rate = 0.0001
module_file_name = "./ayala-module.pth"

t_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = DataLoader(
    dataset=datasets.CIFAR10(
        root="./Dataset",
        train=True,
        transform=transforms.ToTensor(),
        download=True
    ),
    batch_size=64,
    shuffle=True,
    num_workers=0,
    drop_last=True
)
test_loader = DataLoader(
    dataset=datasets.CIFAR10(
        root="./Dataset",
        train=False,
        transform=transforms.ToTensor(),
        download=False
    ),
    batch_size=64,
    shuffle=False,
    num_workers=0,
    drop_last=False
)

writer = SummaryWriter(".\logs")
ayala = Ayala().to(t_device)
loss = nn.CrossEntropyLoss().to(t_device)
opti = optim.SGD(ayala.parameters(), lr=learning_rate)
ayala.load_state_dict(torch.load(module_file_name))

step = 0
step_test = 0
for epoch in range(10):
    # 模型训练过程
    loss_epoch = 0.0
    ayala.train()
    for img, target in train_loader:
        img = img.to(t_device)
        target = target.to(t_device)
        if step % 10 == 0:
            print("|", end="")

        opti.zero_grad()
        output = ayala(img)
        result_loss = loss(output, target)
        result_loss.backward()
        opti.step()

        loss_epoch += result_loss
        step += 1

    # 模型验证过程
    loss_epoch_test = 0.0
    total_accuracy = 0.0
    ayala.eval()
    with torch.no_grad():
        for img, target in test_loader:
            img = img.to(t_device)
            target = target.to(t_device)
            if step_test % 10 == 0:
                print("-", end="")

            output = ayala(img)
            result_loss = loss(output, target)

            accuracy = (output.argmax(1) == target).sum()
            total_accuracy += accuracy
            loss_epoch_test += result_loss
            step_test += 1

    print("\nepoch:", epoch+1, "loss(total,train):%.2f" % loss_epoch,
          "loss(total,test):%.2f" % loss_epoch_test,
          "accuacry(test):%.2f" % (total_accuracy/(len(test_loader)*64)))
    writer.add_scalar("Ayala-loss", loss_epoch, epoch)
    writer.add_scalar("Ayala-test-loss", loss_epoch_test, epoch)
    writer.add_scalar("Ayala-test-accuracy",
                      total_accuracy/(len(test_loader)*64), epoch)
    torch.save(ayala.state_dict(), module_file_name)

writer.close()
