
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.autograd import Variable
from keras.preprocessing.text import Tokenizer
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle
import itertools
from statistics import mean
import matplotlib.pyplot as plt

from transformer_model.turkish_models.bert_model.preprocess import *
from transformer_model.turkish_models.bert_model.mydataloader import MyDataloader
from transformer_model.turkish_models.bert_model.mymodel import MyModel
from transformer_model.turkish_models.bert_model.params import Params

class Main(object):

    def __init__(self):
        pass

    @staticmethod
    def load_dataset():
        data = pd.read_excel(Params.dataset_dir)
        return data

    @staticmethod
    def do_preprocess(data):
        if Params.preprocess_steps["lowercase"]:
            data = lowercase(data)
        if Params.preprocess_steps["remove_punctuations"]:
            data = remove_punctuations(data)
        if Params.preprocess_steps["remove_numbers"]:
            data = remove_numbers(data)
        if Params.preprocess_steps["remove_stop_words"]:
            data = remove_stop_words(data)
        if Params.preprocess_steps["zemberek_stemming"]:
            data = zemberek_stemming(data)  # gives connection error for long documents/dataframes
        if Params.preprocess_steps["first_5_char_stemming"]:
            data = first_5_char_stemming(data)
        if Params.preprocess_steps["data_shuffle"]:
            data = data_shuffle(data, Params.shuffle_count)
        return data

    @staticmethod
    def run_preprocess(data):
        # preprocess
        data = Main.do_preprocess(data)

        X_train, X_test, y_train, y_test = train_test_split(data["text"],
                                                            data["label"],
                                                            test_size=Params.test_split_rate,
                                                            stratify=data["label"],
                                                            shuffle=True,
                                                            random_state=42)
        # del "-", "_" chars from filters
        Params.label_tokenizer = Tokenizer(filters='!"#$%&()*+,./:;<=>?@[\\]^`{|}~\t\n', lower=True)
        Params.label_tokenizer.fit_on_texts(y_train)
        with open(os.path.join(Params.model_dir, Params.label_tokenizer_name), "wb") as handle:
            pickle.dump(Params.label_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        y_train = Params.label_tokenizer.texts_to_sequences(y_train)  # list of list
        y_train = [_y[0] - 1 for _y in y_train] # -1 for start from 0 index
        y_test = Params.label_tokenizer.texts_to_sequences(y_test)
        y_test = [_y[0] - 1 for _y in y_test]  # [batch]

        # convert series to list
        X_train = X_train.tolist()
        X_test = X_test.tolist()

        # create dataloaders
        train_dataloader = DataLoader(dataset=MyDataloader(X_train, y_train),
                                      batch_size=Params.batch_size,
                                      shuffle=Params.batch_shuffle)
        test_dataloader = DataLoader(dataset=MyDataloader(X_test, y_test),
                                     batch_size=Params.batch_size,
                                     shuffle=Params.batch_shuffle)

        return train_dataloader, test_dataloader

    @staticmethod
    def run_train(dataloader):
        Params.model.train()  # set train mode
        epoch_loss = []
        predicted_labels = []
        y_labels = []
        for id, (X, y) in enumerate(dataloader):  # batch
            print("train batch id: ", id)
            encoding = Params.bert_tokenizer(list(X),
                                             return_tensors='pt',
                                             padding=True,
                                             truncation=True,
                                             max_length=Params.bert_sequence_max_length,
                                             add_special_tokens=True)
            #print("encoding:", encoding)
            input_ids = encoding['input_ids']
            token_type_ids = encoding["token_type_ids"]
            attention_mask = encoding['attention_mask']

            # convert tensors to variables
            input_ids = Variable(input_ids, requires_grad=False)
            token_type_ids = Variable(token_type_ids, requires_grad=False)
            attention_mask = Variable(attention_mask, requires_grad=False)
            y = Variable(y, requires_grad=False)

            if Params.use_cuda:
                input_ids = input_ids.cuda()
                token_type_ids = token_type_ids.cuda()
                attention_mask = attention_mask.cuda()
                y = y.cuda()

            outputs = Params.model(input_ids, token_type_ids, attention_mask) # bert
            #print(outputs)
            loss = Params.criterion(outputs, y.long())  # y: long type
            epoch_loss.append(loss.item())  # save batch loss

            # accuracy
            _, predicted = torch.max(outputs.data, 1)
            predicted_labels.append(predicted.tolist())
            y_labels.append(y.tolist())

            # backward and optimize
            Params.optimizer.zero_grad()
            loss.backward()
            Params.optimizer.step()

            torch.cuda.empty_cache()

        pred_flat = list(itertools.chain(*predicted_labels)) # flat list of list to list
        y_flat = list(itertools.chain(*y_labels))
        acc = Main.get_metrics(pred_flat, y_flat)
        return mean(epoch_loss), acc

    @staticmethod
    def run_test(dataloader):
        Params.model.eval()  # set eval mode
        epoch_loss = []
        predicted_labels = []
        y_labels = []
        for id, (X, y) in enumerate(dataloader):  # batch
            encoding = Params.bert_tokenizer(list(X),
                                             return_tensors='pt',
                                             padding=True,
                                             truncation=True,
                                             max_length=Params.bert_sequence_max_length,
                                             add_special_tokens=True)
            # print("encoding:", encoding)
            input_ids = encoding['input_ids']
            token_type_ids = encoding["token_type_ids"]
            attention_mask = encoding['attention_mask']

            # convert tensors to variables
            input_ids = Variable(input_ids, requires_grad=False)
            token_type_ids = Variable(token_type_ids, requires_grad=False)
            attention_mask = Variable(attention_mask, requires_grad=False)
            y = Variable(y, requires_grad=False)

            if Params.use_cuda:
                input_ids = input_ids.cuda()
                token_type_ids = token_type_ids.cuda()
                attention_mask = attention_mask.cuda()
                y = y.cuda()

            outputs = Params.model(input_ids, token_type_ids, attention_mask)  # bert
            # print(outputs)
            loss = Params.criterion(outputs, y.long())  # y: long type
            epoch_loss.append(loss.item())  # save batch loss

            # accuracy
            _, predicted = torch.max(outputs.data, 1)
            predicted_labels.append(predicted.tolist())
            y_labels.append(y.tolist())

            torch.cuda.empty_cache()

        pred_flat = list(itertools.chain(*predicted_labels))  # flat list of list to list
        y_flat = list(itertools.chain(*y_labels))
        acc = Main.get_metrics(pred_flat, y_flat)
        return mean(epoch_loss), acc

    @staticmethod
    def get_metrics(y_pred, y_true):
        acc = accuracy_score(y_true, y_pred)
        return acc

    @staticmethod
    def save_plot(train_acc, train_loss, test_acc, test_loss):
        # loss figure
        plt.clf()
        plt.plot(train_loss, label='train')
        plt.plot(test_loss, label='valid')
        plt.title('train-valid loss')
        plt.ylabel('categorical_crossentropy loss')
        plt.xlabel('epoch')
        plt.legend(loc="upper left")
        # plt.show()
        plt.savefig(os.path.join(Params.plot_dir, "loss.png"))

        # accuracy figure
        plt.clf()
        plt.plot(train_acc, label='train')
        plt.plot(test_acc, label='valid')
        plt.title('train-valid accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(loc="upper left")
        plt.savefig(os.path.join(Params.plot_dir, "accuracy.png"))

    @staticmethod
    def run_train_test(train_dataloader, test_dataloader):
        Params.model = MyModel(bert_model=Params.bert_model,
                               bert_hidden_size=Params.bert_hidden_size,
                               hidden_1_size=Params.hidden_1_size,
                               hidden_2_size=Params.hidden_2_size,
                               class_size=len(Params.label_tokenizer.word_index))
        # push model to gpu
        if Params.use_cuda:
            Params.model = Params.model.cuda()

        pytorch_total_params = sum(p.numel() for p in Params.model.parameters())
        print("pytorch_total_params: ", pytorch_total_params)

        Params.optimizer = torch.optim.Adam(Params.model.parameters(), lr=Params.lr)
        Params.criterion = nn.CrossEntropyLoss()

        train_loss = []
        test_loss = []
        train_acc = []
        test_acc = []
        for epoch in range(1, Params.epoch + 1):
            print(epoch, " .epoch başladı ...")
            # train
            _train_loss, _train_acc = Main.run_train(train_dataloader)
            train_loss.append(_train_loss)
            train_acc.append(_train_acc)

            # test
            _test_loss, _test_acc = Main.run_test(test_dataloader)
            test_loss.append(_test_loss)
            test_acc.append(_test_acc)

            # info
            print("train loss -> ", _train_loss)
            print("train acc -> ", _train_acc)

            print("test loss -> ", _test_loss)
            print("test acc -> ", _test_acc)

        # plot
        Main.save_plot(train_acc, train_loss, test_acc, test_loss)

if __name__ == '__main__':

    print("cuda available: ", torch.cuda.is_available())

    data = Main.load_dataset()
    train_dataloader, test_dataloader = Main.run_preprocess(data)
    Main.run_train_test(train_dataloader, test_dataloader)

    """
        - keras gib lr i azaltan bir yapı yaz ...
        - fully connected sonra drop ekle ...
    """

