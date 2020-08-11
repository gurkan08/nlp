
import numpy as np
import keras as k
from keras.callbacks import *
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import matplotlib.pyplot as plt

from network import Network

class Main(object):
    train_data_dir = "MIT_Movie_Corpus/engtrain.bio"
    test_data_dir = "MIT_Movie_Corpus/engtest.bio"

    word2id = {}
    id2word = {}
    tag2id = {}
    id2tag = {}

    max_sentence_len = -100000000  # kelime bazında
    min_sentence_len = 100000000  # kelime bazında
    padding_word = "padding_word"  # not used
    empty_tag = "O".lower()  # bunu datasete göre güncelle !, tag leri padding yaparken bunu kullan :)

    # network params = network spesific params
    epoch = 1
    batch_size = 20
    validation_split_rate = 0.1
    shuffle_batch = True
    embedding_dim = 50
    lstm_hidden_dim = 50
    num_layers = 1
    lr = 0.00025
    padding_type = 'post'
    padding_value = 0.
    early_stop_patience = 2


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
    def convert2numeric(X, y):
        X_all = []
        y_all = []
        for X, y in zip(X, y):
            _X = []
            _y = []
            for word in X.split():
                _X.append(int(Main.word2id[word]))
            X_all.append(_X)

            for tag in y.split():
                _y.append(Main.tag2id[tag])
            y_all.append(_y)
        return X_all, y_all

    @staticmethod
    def padding(X, y):
        X = pad_sequences(np.array(X),
                          maxlen=Main.max_sentence_len,
                          dtype='int32',
                          padding=Main.padding_type,
                          value=Main.padding_value)
        y = pad_sequences(np.array(y),
                          maxlen=Main.max_sentence_len,
                          dtype='int32',
                          padding=Main.padding_type,
                          value=Main.padding_value)
        y = [to_categorical(i, num_classes=len(Main.tag2id)) for i in y]  # BU SATIR OLMADAN ÇALIŞMIYOR !
        return X, y

    @staticmethod
    def plot_info(history, name="loss"):
        # to prevent overlap, clear the buffer
        plt.clf()

        if name=="loss":
            plt.plot([i for i in range(1, Main.epoch + 1)], history["loss"])
            plt.plot([i for i in range(1, Main.epoch + 1)], history["val_loss"])
            plt.xlabel("epoch")
            plt.ylabel(name)
            plt.title(name + " graphic")
            plt.legend(["loss", "val_loss"])
            # plt.show()
            plt.savefig(name + ".png")

        elif name=="acc":
            plt.plot([i for i in range(1, Main.epoch + 1)], history["acc"])
            plt.plot([i for i in range(1, Main.epoch + 1)], history["crf_viterbi_accuracy"])
            plt.plot([i for i in range(1, Main.epoch + 1)], history["val_acc"])
            plt.plot([i for i in range(1, Main.epoch + 1)], history["val_crf_viterbi_accuracy"])
            plt.xlabel("epoch")
            plt.ylabel(name)
            plt.title(name + " graphic")
            plt.legend(["acc", "crf_viterbi_accuracy", "val_acc", "val_crf_viterbi_accuracy"])
            # plt.show()
            plt.savefig(name + ".png")

    @staticmethod
    def run_pipeline():
        X_train_sentences, X_train_tags, X_test_sentences, X_test_tags = Main.load_dataset()
        Main.create_dicts(X_train_sentences, X_train_tags, X_test_sentences, X_test_tags)
        # convert to numeric format
        X_train, y_train = Main.convert2numeric(X_train_sentences, X_train_tags)
        X_test, y_test = Main.convert2numeric(X_test_sentences, X_test_tags)
        # padding
        X_train, y_train = Main.padding(X_train, y_train)
        X_test, y_test = Main.padding(X_test, y_test)

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

        # create optimizer, model, loss
        optimizer = k.optimizers.Adam(lr=Main.lr, beta_1=0.9, beta_2=0.999)
        model = Network(max_sentence_len=Main.max_sentence_len,
                        vocab_size=len(Main.word2id),
                        embed_dim=Main.embedding_dim,
                        tag_size=len(Main.tag2id),
                        optimizer=optimizer).get_model()
        print("-----------model summary-----------------> ")
        print(model.count_params())
        print(model.summary())
        print("-----------model summary-----------------> ")

        """
        # Saving the best model only
        model_path = "ner-model-keras-{val_accuracy:.2f}.hdf5"
        checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        """

        my_callbacks = [
            EarlyStopping(patience=Main.early_stop_patience),
            ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5',
                            monitor='val_loss',
                            save_best_only=True,
                            mode='max'),
            TensorBoard(log_dir='./logs'),
        ]

        # train
        history = model.fit(np.array(X_train),
                            np.array(y_train),
                            batch_size=Main.batch_size,
                            epochs=Main.epoch,
                            validation_split=Main.validation_split_rate,
                            verbose=1,
                            callbacks=my_callbacks)
        # print(history.history) # get acc, loss info
        # plot loss, accuracy
        Main.plot_info(history.history, name="loss")
        Main.plot_info(history.history, name="acc")

        # test
        text = "what movies star bruce willis"
        text_numeric = []
        for word in text.split():
            text_numeric.append(Main.word2id[word])
        # padding
        for i in range(len(text_numeric), Main.max_sentence_len):
            text_numeric.append(0)
        text_numeric = np.array(text_numeric).reshape((-1, 1)) # (max_sentence_len, ) formatında olmalı
        #print(text_numeric)
        print(text_numeric.shape)
        exit()

        result = model.predict(text_numeric)
        print(result)

        exit()


        # history accuraccy hesapla grafik bastır



if __name__ == '__main__':

    Main.run_pipeline()

    """
        - acc/epoch grafik çizdir
        - loss/epoch grafik çizdir
        - model.predict() kısmını da kodla, 1-2 sonucu al bajk doğru mu :)
        - rest api haline getir postman sorgu at :))
    """







