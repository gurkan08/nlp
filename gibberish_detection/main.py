
import itertools
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch
from statistics import mean
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from mydataloader import MyDataloader
from network import Network

class Main(object):

    nongibberish_data_dir = "kaggle_gibberish_dataset/nongibberish.txt"
    gibberish_data_dir = "kaggle_gibberish_dataset/gibberish.csv"
    nongibberish_label = 0
    gibberish_label = 1
    test_size = 0.25
    num_class = None
    max_gibberish_sample_size = 3700 # set manually, to balance dataset
    loss_figure_dir = "loss.png"
    acc_figure_dir = "acc.png"

    # nn spesific params
    epoch = 50
    batch_size = 25
    shuffle_batch = True
    lr = 0.00025
    use_cuda = torch.cuda.is_available()

    def __init__(self):
        pass

    @staticmethod
    def load_dataset():
        X = []
        y = []

        # nongibberish
        with open(Main.nongibberish_data_dir) as fp:
            lines = fp.readlines()
        for line in lines:
            line = line.lower().strip()
            if line != "":
                X.append(line)
                y.append(0)
                # to balanced dataset
                if len(X) == Main.max_gibberish_sample_size:
                    break

        # gibberish
        with open(Main.gibberish_data_dir) as fp:
            lines = fp.readlines()
        for line in lines:
            line = line.lower().strip().replace(",1", "") # gibberish file spesific
            if line != "":
                X.append(line)
                y.append(1)

        return X, y

    @staticmethod
    def get_dataloader(X, y):
        dataloader = DataLoader(dataset=MyDataloader(X, y),
                                batch_size=Main.batch_size,
                                shuffle=Main.shuffle_batch)
        return dataloader

    @staticmethod
    def get_metrics(pred, y):
        """
        positive -> 1 (gibberish)
        negative -> 0 (nongibberish)
        """
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        for _pred, _y in zip(pred, y):
            # tp
            if _pred==1 and _y==1:
                tp += 1
            # fp
            if _pred==1 and _y==0:
                fp += 1
            # fn
            if _pred==0 and _y==1:
                fn += 1
            # tn
            if _pred==0 and _y==0:
                tn += 1

        sum_positive = tp + fn
        sum_negative = fp + tn
        tp_rate = float(tp) / (sum_positive + 1)
        fp_rate = float(fp) / (sum_negative + 1)
        acc = float(tp + tn) / (tp + fp + fn + tn)
        return tp_rate, fp_rate, acc

    @staticmethod
    def run_train(dataloader, model, criterion, optimizer):
        model.train()  # set train mode
        epoch_loss = []
        predicted_labels = []
        y_labels = []
        for id, (X, y) in enumerate(dataloader):  # batch
            # convert tensors to variables
            X = Variable(X, requires_grad=False)
            y = Variable(y, requires_grad=False)
            if Main.use_cuda:
                X = X.cuda()
                y = y.cuda()
                model = model.cuda()

            outputs = model(X.float())  # float type
            loss = criterion(outputs, y.long())  # y: long type
            epoch_loss.append(loss.item())  # save batch loss

            # accuracy
            _, predicted = torch.max(outputs.data, 1)
            predicted_labels.append(predicted.tolist())
            y_labels.append(y.tolist())

            # backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        pred_flat = list(itertools.chain(*predicted_labels)) # flat list of list to list
        y_flat = list(itertools.chain(*y_labels))
        tp_rate, fp_rate, acc = Main.get_metrics(pred_flat, y_flat)
        return mean(epoch_loss), tp_rate, fp_rate, acc

    @staticmethod
    def run_test(dataloader, model, criterion):
        model.eval()  # set eval mode
        epoch_loss = []
        predicted_labels = []
        y_labels = []
        for id, (X, y) in enumerate(dataloader):  # batch
            # convert tensors to variables
            X = Variable(X, requires_grad=False)
            y = Variable(y, requires_grad=False)
            if Main.use_cuda:
                X = X.cuda()
                y = y.cuda()
                model = model.cuda()

            outputs = model(X.float())  # float type
            loss = criterion(outputs, y.long())  # y: long type
            epoch_loss.append(loss.item())  # save batch loss

            # accuracy
            _, predicted = torch.max(outputs.data, 1)
            predicted_labels.append(predicted.tolist())
            y_labels.append(y.tolist())

        pred_flat = list(itertools.chain(*predicted_labels))  # flat list of list to list
        y_flat = list(itertools.chain(*y_labels))
        tp_rate, fp_rate, acc = Main.get_metrics(pred_flat, y_flat)
        return mean(epoch_loss), tp_rate, fp_rate, acc

    @staticmethod
    def plot_loss_figure(train_loss, test_loss):
        # to prevent overlap, clear the buffer
        plt.clf()

        plt.plot([i for i in range(1, Main.epoch + 1)], train_loss)
        plt.plot([i for i in range(1, Main.epoch + 1)], test_loss)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("loss graphic")
        plt.legend(["train", "test"])
        # plt.show()
        plt.savefig(Main.loss_figure_dir)

    @staticmethod
    def plot_acc_figure(train_acc, test_acc):
        # to prevent overlap, clear the buffer
        plt.clf()

        plt.plot([i for i in range(1, Main.epoch + 1)], train_acc)
        plt.plot([i for i in range(1, Main.epoch + 1)], test_acc)
        plt.xlabel("epoch")
        plt.ylabel("acc")
        plt.title("acc graphic")
        plt.legend(["train", "test"])
        # plt.show()
        plt.savefig(Main.acc_figure_dir)

    @staticmethod
    def run_pipeline():
        X, y = Main.load_dataset()
        Main.num_class = len(set(y))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Main.test_size, random_state=39)
        train_dataloader = Main.get_dataloader(X_train, y_train)
        test_dataloader = Main.get_dataloader(X_test, y_test)

        # model, loss, optimizer
        model = Network(num_class=Main.num_class)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=Main.lr)

        train_loss = []
        test_loss = []
        train_acc = []
        test_acc = []
        for epoch in range(1, Main.epoch + 1):
            print(epoch, " .epoch başladı ...")
            # train
            train_epoch_loss, _train_tp_rate, _train_fp_rate, _train_acc = Main.run_train(train_dataloader, model, criterion, optimizer)
            train_loss.append(train_epoch_loss)
            train_acc.append(_train_acc)

            # test
            test_epoch_loss, _test_tp_rate, _test_fp_rate, _test_acc = Main.run_test(test_dataloader, model, criterion)
            test_loss.append(test_epoch_loss)
            test_acc.append(_test_acc)

            # info
            print("train loss -> ", train_epoch_loss)
            print("train tpr -> ", _train_tp_rate)
            print("train fpr -> ", _train_fp_rate)
            print("train acc -> ", _train_acc)

            print("test loss -> ", test_epoch_loss)
            print("test tpr -> ", _test_tp_rate)
            print("test fpr -> ", _test_fp_rate)
            print("test acc -> ", _test_acc)

            print(epoch, " .epoch bitti ...")

        # plot loss
        Main.plot_loss_figure(train_loss, test_loss)
        Main.plot_acc_figure(train_acc, test_acc)



if __name__ == '__main__':

    Main.run_pipeline()

