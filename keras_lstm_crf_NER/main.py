
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
from statistics import mean
import matplotlib.pyplot as plt
import numpy as np
import keras as k
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from keras_ner_lstm_crf.mydataloader import MyDataloader
from keras_ner_lstm_crf.network import Network

class Main(object):
    train_data_dir = "MIT_Movie_Corpus/engtrain.bio"
    test_data_dir = "MIT_Movie_Corpus/engtest.bio"
    loss_figure_dir = "loss.png"

    word2id = {}
    id2word = {}
    tag2id = {}
    id2tag = {}

    max_sentence_len = -100000000  # kelime bazında
    min_sentence_len = 100000000  # kelime bazında
    padding_word = "padding_word"  # not used
    empty_tag = "O".lower()  # bunu datasete göre güncelle !, tag leri padding yaparken bunu kullan :)

    # network params = network spesific params
    epoch = 250
    batch_size = 20
    validation_split_rate = 0.1
    shuffle_batch = True
    embedding_dim = 50
    lstm_hidden_dim = 50
    num_layers = 1
    lr = 0.00025
    use_cuda = torch.cuda.is_available()

    def __init__(self):
        pass

    @staticmethod
    def load_dataset():
        """
        :return: X_train_sentences -> list of string, X_train_tags -> list of string
                X_test_sentences -> list of string, X_test_tags -> list of string
        """
        # train file
        X_train_sentences = []
        X_train_tags = []
        with open(Main.train_data_dir) as fp:
            lines = fp.readlines()

        words = []
        tags = []
        for line in lines:
            # print(line.strip())
            line = line.strip().lower().split()  # lowercase
            if len(line) != 0:  # cümleler arası boşluk kontrolü
                words.append(line[1])
                tags.append(line[0])
            elif len(line) == 0:  # yeni cümleye geçtik
                X_train_sentences.append(" ".join(words))
                X_train_tags.append(" ".join(tags))

                if len(words) > Main.max_sentence_len:
                    Main.max_sentence_len = len(words)
                if len(words) < Main.min_sentence_len:
                    Main.min_sentence_len = len(words)

                # reset
                words = []
                tags = []

        # test file
        X_test_sentences = []
        X_test_tags = []
        with open(Main.test_data_dir) as fp:
            lines = fp.readlines()

        words = []
        tags = []
        for line in lines:
            # print(line.strip())
            line = line.strip().lower().split()  # lowercase
            if len(line) != 0:  # cümleler arası boşluk kontrolü
                words.append(line[1])
                tags.append(line[0])
            elif len(line) == 0:  # yeni cümleye geçtik
                X_test_sentences.append(" ".join(words))
                X_test_tags.append(" ".join(tags))

                if len(words) > Main.max_sentence_len:
                    Main.max_sentence_len = len(words)
                if len(words) < Main.min_sentence_len:
                    Main.min_sentence_len = len(words)

                # reset
                words = []
                tags = []

        return X_train_sentences, X_train_tags, X_test_sentences, X_test_tags

    @staticmethod
    def create_dicts(X_train_sentences, X_train_tags, X_test_sentences, X_test_tags):
        """
        :param X_train_sentences: list of string
        :param X_train_tags: list of string
        :param X_test_sentences: list of string
        :param X_test_tags: list of string
        :return: None
        """
        vocab_id = 1  # 0 for oov word :)
        tag_id = 0

        # train samples
        for (sentence, tags) in zip(X_train_sentences, X_train_tags):
            # print(sentence, " --> ", tags)
            sentence = sentence.lower().strip()
            tags = tags.lower().strip()

            for word in sentence.split():
                if word not in Main.word2id:
                    Main.word2id[word] = vocab_id
                    Main.id2word[str(vocab_id)] = word
                    vocab_id += 1

            for tag in tags.split():
                if tag not in Main.tag2id:
                    Main.tag2id[tag] = tag_id
                    Main.id2tag[str(tag_id)] = tag
                    tag_id += 1

        # test samples
        for (sentence, tags) in zip(X_test_sentences, X_test_tags):
            # print(sentence, " --> ", tags)
            sentence = sentence.lower().strip()
            tags = tags.lower().strip()

            for word in sentence.split():
                if word not in Main.word2id:
                    Main.word2id[word] = vocab_id
                    Main.id2word[str(vocab_id)] = word
                    vocab_id += 1

            for tag in tags.split():
                if tag not in Main.tag2id:
                    Main.tag2id[tag] = tag_id
                    Main.id2tag[str(tag_id)] = tag
                    tag_id += 1

    @staticmethod
    def get_dataloader(X, y):
        """
        :param X: list of string
        :param y: list of string
        :return: dataloader obj
        """
        dataloader = DataLoader(dataset=MyDataloader(X, y,
                                                     Main.word2id,
                                                     Main.tag2id,
                                                     Main.max_sentence_len,
                                                     Main.empty_tag),
                                batch_size=Main.batch_size,
                                shuffle=Main.shuffle_batch)
        return dataloader

    @staticmethod
    def run_train(dataloader, model, optimizer):
        model.train()  # set train mode
        epoch_loss = []
        for id, (X, y, len_X) in enumerate(dataloader):  # batch
            # convert tensors to variables
            X = Variable(X, requires_grad=False)
            y = Variable(y, requires_grad=False)
            if Main.use_cuda:
                X = X.cuda()
                y = y.cuda()
                model = model.cuda()

            # forward pass
            batch_len = len_X.squeeze().tolist()  # list: word/sequences length of each batch elements
            loss, crf_decode_result = model(X.float(), y.long(), batch_len)  # float type
            # print("loss ---> ", loss.item())
            epoch_loss.append(loss.item())

            # backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return mean(epoch_loss), crf_decode_result

    @staticmethod
    def run_test(dataloader, model):
        model.eval()  # set eval mode
        epoch_loss = []
        for id, (X, y, len_X) in enumerate(dataloader):  # batch
            # convert tensors to variables
            X = Variable(X, requires_grad=False)
            y = Variable(y, requires_grad=False)
            if Main.use_cuda:
                X = X.cuda()
                y = y.cuda()
                model = model.cuda()

            # forward pass
            batch_len = len_X.squeeze().tolist()  # list: word/sequences length of each batch elements
            loss, crf_decode_result = model(X.float(), y.long(), batch_len)  # float type
            # print("loss ---> ", loss.item())
            epoch_loss.append(loss.item())
        return mean(epoch_loss), crf_decode_result

    @staticmethod
    def plot_loss_figure(train_loss, test_loss):
        plt.plot([i for i in range(1, Main.epoch + 1)], train_loss, label="train")
        plt.plot([i for i in range(1, Main.epoch + 1)], test_loss, label="test")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("loss graphic")
        # plt.show()
        plt.savefig(Main.loss_figure_dir)

    @staticmethod
    def run_pipeline():
        X_train_sentences, X_train_tags, X_test_sentences, X_test_tags = Main.load_dataset()
        Main.create_dicts(X_train_sentences, X_train_tags, X_test_sentences, X_test_tags)

        """
        print(Main.word2id)
        print(Main.id2word)
        print(Main.tag2id)
        print(Main.id2tag)
        print(len(Main.word2id))
        print(len(Main.tag2id))
        print(Main.max_sentence_len) # 47
        print(Main.min_sentence_len) # 1
        """

        train_dataloader = Main.get_dataloader(X_train_sentences, X_train_tags)
        test_dataloader = Main.get_dataloader(X_test_sentences, X_test_tags)

        # optimizer, model, loss
        optimizer = k.optimizers.Adam(lr=Main.lr, beta_1=0.9, beta_2=0.999)
        model = Network(max_sentence_len=Main.max_sentence_len,
                        vocab_size=len(Main.word2id),
                        embed_dim=Main.embedding_dim,
                        tag_size=len(Main.tag2id),
                        optimizer=optimizer)

        model = model.get_model() # önemli !
        print("-----------model summary-----------------> ")
        print(model.count_params())
        print(model.summary())
        print("-----------model summary-----------------> ")

        # Saving the best model only
        model_path = "ner-model-keras-{val_accuracy:.2f}.hdf5"
        checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]



        # train

        # convert string to numeric type
        X_numeric = []
        y_numeric = []
        for X, y in zip(X_train_sentences, X_train_tags):
            _X = []
            _y = []
            for word in X.split():
                _X.append(int(Main.word2id[word]))
            X_numeric.append(_X)

            for tag in y.split():
                _y.append(Main.tag2id[tag])
            y_numeric.append(_y)

        # padding
        X_train_sentences = pad_sequences(X_numeric,
                                          maxlen=Main.max_sentence_len,
                                          dtype='int32',
                                          padding='post',
                                          value=0.0)

        y_train_sentences = pad_sequences(y_numeric,
                                          maxlen=Main.max_sentence_len,
                                          dtype='int32',
                                          padding='post',
                                          value=0.0)

        y_train_sentences = [to_categorical(i, num_classes=len(Main.tag2id)) for i in y_train_sentences] # BU SATIR OLMADAN ÇALIŞMIYOR !

        history = model.fit(np.array(X_train_sentences),
                            np.array(y_train_sentences),
                            batch_size=Main.batch_size,
                            epochs=Main.epoch,
                            validation_split=Main.validation_split_rate,
                            verbose=1,
                            callbacks=callbacks_list)

        #model.save('model_weights.h5')
        model.save(model_path)
        #return history, model




if __name__ == '__main__':

    Main.run_pipeline()

    """
        - acc/epoch grafik çizdir
        - loss/epoch grafik çizdir
        - model.predict() kısmını da kodla, 1-2 sonucu al bajk doğru mu :)
        - rest api haline getir postman sorgu at :))
    """







