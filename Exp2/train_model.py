import torch
import random
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split

from model import Data, CNN_net

class Lab2Model(object):

    def __init__(self, batch_size=64, num_workers=10, seed=0):
        self.seed = seed
        self.setup_seed()
        # 这里 ToTensor 会把 numpy 类型转换为 tensor 类型，并对数据归一化到 [0, 1]
        train_dataset = Data(type_="train", transform=transforms.Compose([transforms.ToTensor()]))

        # 从训练数据中手动划分训练集和验证集
        self.train_dataset, self.val_dataset = random_split(train_dataset,
                                                            [int(len(train_dataset) * 0.8),
                                                             len(train_dataset) - int(len(train_dataset) * 0.8)],
                                                            generator=torch.Generator().manual_seed(0))

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=num_workers,
                                           drop_last=True)
        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=num_workers,
                                         drop_last=True)

        self.net = None
        self.lr = None
        self.optimizer = None
        self.device = None
        self.schedule = None
        self.fig_name = None
        self.loss_list = {"train": [], "val": []}
        self.acc_list = {"train": [], "val": []}

    def train(self, lr=0.01, epochs=10, device="cuda", wait=8, lrd=False, fig_name="lab1"):
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")
        self.lr = lr
        self.fig_name = fig_name
        self.net = CNN_net(normalize=True).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        if lrd:
            self.schedule = ReduceLROnPlateau(self.optimizer, 'min', patience=1, verbose=True)

        total_params = sum([param.nelement() for param in self.net.parameters() if param.requires_grad])

        print(">>> Total params: {}".format(total_params))

        print(">>> Start training")
        min_val_loss = np.inf
        min_val_loss_acc = 0.0
        delay = 0
        for epoch in range(epochs):

            # train train data
            for data in tqdm(self.train_dataloader):
                labels, inputs = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                loss.backward()
                self.optimizer.step()

            # calc train loss and train acc
            train_loss = 0.0
            train_acc = 0.0
            val_loss = 0.0
            val_acc = 0.0
            with torch.no_grad():
                for data in self.train_dataloader:
                    labels, inputs = data
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.net(inputs)
                    loss = nn.CrossEntropyLoss()(outputs, labels)

                    train_loss += loss.item()
                    train_acc += self.acc(labels=labels.cpu().numpy(), outputs=outputs.detach().cpu().numpy())
                train_loss = train_loss / len(self.train_dataloader)
                train_acc = train_acc / len(self.train_dataloader)

                self.loss_list['train'].append(train_loss)
                self.acc_list['train'].append(train_acc)

                for data in self.val_dataloader:
                    labels, inputs = data
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.net(inputs)
                    loss = nn.CrossEntropyLoss()(outputs, labels)

                    val_loss += loss.item()
                    val_acc += self.acc(labels=labels.cpu().numpy(), outputs=outputs.detach().cpu().numpy())
                val_loss = val_loss / len(self.val_dataloader)
                val_acc = val_acc / len(self.val_dataloader)
                self.loss_list['val'].append(val_loss)
                self.acc_list['val'].append(val_acc)
                print(f"Epoch {epoch}: train loss {train_loss:10.6f}, acc {train_acc:7.4f}, "
                      f"val loss {val_loss:10.6f}, acc {val_acc:7.4f}, ")
            # if necessary, reduce the learning rate by val loss
            if lrd:
                self.schedule.step(val_loss)

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                min_val_loss_acc = val_acc
                print(f"Update min_val_loss to {min_val_loss:10.6f}")
                delay = 0
            else:
                delay = delay + 1

            if delay > wait:
                break
        print(">>> Finished training")
        self.plot_loss()
        self.plot_acc()
        print(">>> Finished plot loss")
        return min_val_loss_acc

    def test(self):
        test_data = Data(type_="val", transform=transforms.Compose([transforms.ToTensor()]))
        test_data_loader = DataLoader(dataset=test_data,
                                      batch_size=self.batch_size,
                                      shuffle=False,
                                      num_workers=self.num_workers,
                                      drop_last=False)

        test_acc = 0.0
        for data in test_data_loader:
            labels, inputs = data
            inputs = inputs.to(self.device)
            outputs = self.net(inputs)
            test_acc += self.acc(labels.numpy(), outputs.detach().cpu().numpy())

        test_acc = test_acc / len(test_data_loader)
        return test_acc

    def acc(self, labels, outputs, type_="top1"):
        acc = 0
        if type_ == "top1":
            pre_labels = np.argmax(outputs, axis=1)
            labels = labels.reshape(len(labels))
            acc = np.sum(pre_labels == labels) / len(pre_labels)

        return acc

    def setup_seed(self):
        seed = self.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)

    def plot_loss(self):
        plt.figure()
        train_loss = self.loss_list['train']
        val_loss = self.loss_list['val']
        plt.plot(train_loss, c="red", label="train_loss")
        plt.plot(val_loss, c="blue", label="val_loss")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("CrossEntropyLoss")
        plt.title("CrossEntropyLoss of Train and Validation in each Epoch")
        plt.savefig(f"fig/{self.fig_name}_loss.png")

    def plot_acc(self):
        plt.figure()
        train_acc = self.acc_list['train']
        val_acc = self.acc_list['val']
        plt.plot(train_acc, c="red", label="train_acc")
        plt.plot(val_acc, c="blue", label="val_acc")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy of Train and Validation in each Epoch")
        plt.savefig(f"fig/{self.fig_name}_acc.png")