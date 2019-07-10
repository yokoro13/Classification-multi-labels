import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from data_loader import get_loader
import time
from Net import Net
import os


def classification_loss(logit, target):
    return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = get_loader(image_dir="./data/celeba/images/", attr_path="./data/celeba/list_attr_celeba.txt",
                            selected_attrs=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'])

    data_iter = iter(dataloader)
    data_img, data_label = next(data_iter)
    data_img, data_label = data_img.to(device), data_label.to(device)

    num_iters = 200000

    start_time = time.time()

    net = Net()
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    c_dim = 5

    running_loss = 0
    for i in range(num_iters):
        try:
            data_img, data_label = next(data_iter)
        except:
            data_iter = iter(dataloader)
            data_img, data_label = next(data_iter)

        data_img, data_label = data_img.to(device), data_label.to(device)
        optimizer.zero_grad()

        out_cls = net(data_img)
        out_cls = out_cls[:, :c_dim]

        loss = classification_loss(out_cls, data_label)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (i+1) % 100 == 0:
            print('[{:d}, loss: {}'.format(i+1, running_loss/100))
            running_loss = 0

        if (i+1) % 1000 == 0:
            print("data_label" + str(data_label) + "out_cls" + str(out_cls))

        if (i+1) % 10000 == 0:
            g_path = os.path.join("./models/", "{}.ckpt".format(i+1))
            torch.save(net.state_dict(), g_path)
            print("saving {}.ckpt...".format(i+1))
