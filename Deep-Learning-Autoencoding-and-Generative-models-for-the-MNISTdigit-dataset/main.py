import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

BATCH_SIZE = 64
LR = 0.001
NUM_OF_EPOCHES = 10
LATENT_DIM = 10
MLP_TRAIN_SIZE = 10


def get_dataset(batch=BATCH_SIZE):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Pad(2),
         transforms.Normalize((0.5), (0.5))])

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch, shuffle=True)

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

    return trainloader, testloader


# class Encoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(32, 16, (3, 3), stride=(2, 2))
#         self.act1 = nn.ReLU()
#         self.conv2 = nn.Conv2d(16, 8, (3, 3), stride=(2, 2))
#         self.act2 = nn.ReLU()
#         self.fc1 = nn.Linear(8 * 2 * 2, 16)
#         self.fc2 = nn.Linear(16, 12)
#         self.fc3 = nn.Linear(12, 8)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.act1(x)
#         x = self.conv2(x)
#         x = self.act2(x)
#         x = torch.flatten(x, 1)  # flatten all dimensions except batch
#         x = self.fc1(x)
#         x = self.fc2(x)
#         x = self.fc3(x)
#         return x
#
#
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc = nn.Sequential(nn.Linear(LATENT_DIM, 128),
                                nn.ReLU(True),
                                nn.Linear(128, 192),)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(48, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 6, stride=2, padding=1),  # b, 1, 32, 32
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = self.decoder(x.reshape(x.shape[0], 48, 2, 2))
        return x





class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=2, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(64, 48, 2, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 8, 2, 2
            nn.Flatten(1),
            nn.Linear(192, 128),
            nn.ReLU(True),
            nn.Linear(128, LATENT_DIM),
        )

        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x

    def get_latent_vector(self):
        return self.encoder[-1].weight.detach()


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 32, 32)
    return x


def train_nn(nn_net, trainloader, testloader, criterion, optimizer, net_name):
    train_loss, test_loss = [], []
    for epoch in range(NUM_OF_EPOCHES):  # loop over the dataset multiple times

        running_loss = 0.0
        # running_acc = 0.0
        counter = 0
        for i, data in enumerate(trainloader, 0):
            counter += 1
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = nn_net(inputs)
            if net_name in ["MLP", "ENCODER+MLP"]:
                loss = criterion(outputs.flatten(), labels.float())
            else:
                loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # running_acc += (inputs.eq(outputs.long())).sum().item() / (inputs.shape[0] * 32 * 32)
            # if i % BATCH_SIZE == BATCH_SIZE:    # print every 2000 mini-batches
            if net_name in ["MLP", "ENCODER+MLP"]:
                break
        train_loss.append(running_loss / counter)
        # train_acc.append(running_acc / counter)

        running_loss = 0.0
        running_acc = 0.0
        counter = 0

        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in testloader:
                counter += 1
                images, labels = data
                # calculate outputs by running images through the network
                outputs = nn_net(images)
                # the class with the highest energy is what we choose as prediction
                # _, predicted = torch.max(outputs.data, 1)
                # total += inputs.size(0)
                # correct += (predicted == labels).sum().item()
                if net_name in ["MLP", "ENCODER+MLP"]:
                    loss = criterion(outputs.flatten(), labels.float())
                else:
                    loss = criterion(outputs, images)
                running_loss += loss.item()
                # running_acc += (images.eq(outputs.long())).sum().item() / (images.shape[0] * 32 * 32)
                # if counter == 1:
                #     pic1 = to_img(outputs.cpu().data)
                #     pic2 = to_img(images.cpu().data)
                #     torchvision.utils.save_image(pic1, f'./results/got_{epoch}.png')
                #     torchvision.utils.save_image(pic2, f'./results/original_{epoch}.png')
        test_loss.append(running_loss / counter)
        # test_acc.append(running_acc / counter)

        # print('Accuracy of the network on the 10000 test images: %d %%' % (
        #         100 * correct / total))


    plt.title(net_name)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(train_loss, label=f'train loss')
    plt.plot(test_loss, label=f'test loss')
    plt.legend()
    plt.show()

net = AutoEncoder()
crit = nn.MSELoss()
opt = optim.Adam(net.parameters(), lr=LR)

tr_loader, tst_loader = get_dataset()

# train_acc, test_acc = [], []
train_nn(net, tr_loader, tst_loader, crit, opt, "AE")

latent_weights = net.get_latent_vector()
pearson_corr = torch.corrcoef(latent_weights)
corr = torch.mean(torch.abs(pearson_corr)).item()
print(f'While Latent dim = {LATENT_DIM}, Pearson Corr (abs + mean) = {corr}')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.plot(train_acc, label=f'train acc')
# plt.plot(test_acc, label=f'test acc')
# plt.legend()
# plt.show()


print('Finished Training')

plt.xlabel('Latent Dimension')
plt.ylabel('Pearson Correlation')
plt.stem([5, 10, 15, 20], [0.298, 0.194, 0.153, 0.136])
plt.xticks([5, 10, 15, 20])
plt.show()



############################# Q3 #######################################

class MLP_AE(nn.Module):
    def __init__(self, encoder):
        super(MLP_AE, self).__init__()
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.mlp = nn.Sequential(nn.Linear(LATENT_DIM, 1),
                                 nn.Softmax())

    def forward(self, x):
        x = self.encoder(x)
        x = self.mlp(x)
        return x

    def train_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True


encoder = net.encoder
mlp_net = MLP_AE(encoder)
tr_loader, tst_loader = get_dataset(MLP_TRAIN_SIZE)
crit = nn.NLLLoss()
opt = optim.Adam(net.parameters(), lr=LR)
train_nn(mlp_net, tr_loader, tst_loader, crit, opt, "MLP")


encoder = net.encoder
mlp_net = MLP_AE(encoder)
tr_loader, tst_loader = get_dataset(MLP_TRAIN_SIZE)
crit = nn.NLLLoss()
opt = optim.Adam(net.parameters(), lr=LR)
mlp_net.train_encoder()
train_nn(mlp_net, tr_loader, tst_loader, crit, opt, "ENCODER+MLP")








