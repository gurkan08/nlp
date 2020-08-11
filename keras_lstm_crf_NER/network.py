
import keras as k
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF

class Network(object):

    lstm_drop_rate = 0.5
    lstm_recurrent_drop_rate = 0.5
    lstm_out_space = 50

    def __init__(self, max_sentence_len, vocab_size, embed_dim, tag_size, optimizer):
        """
        :param max_sentence_len:
        :param vocab_size:
        :param embed_dim:
        :param tag_size:
        """
        self.max_sentence_len = max_sentence_len
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.tag_size = tag_size
        self.optimizer = optimizer
        self.model = self.create_model()

    def get_model(self):
        return self.model

    def create_model(self):
        """
        :return: model
        """
        input = Input(shape=(self.max_sentence_len,))

        # set layers
        output = Embedding(input_dim=self.vocab_size,
                           output_dim=self.embed_dim,
                           input_length=self.max_sentence_len)(input)
        output = Bidirectional(LSTM(units=Network.lstm_out_space,
                                    return_sequences=True,
                                    dropout=Network.lstm_drop_rate,
                                    recurrent_dropout=Network.lstm_recurrent_drop_rate,
                                    kernel_initializer=k.initializers.he_normal()))(output)
        output = TimeDistributed(Dense(self.tag_size, activation="relu"))(output)
        # output = CRF(tag_size)(output)
        crf = CRF(self.tag_size)
        output = crf(output)

        model = Model(input, output)
        model.compile(optimizer=self.optimizer, loss=crf.loss_function, metrics=[crf.accuracy, 'accuracy'])
        #print("------------model summary---------------")
        #model.summary()

        return model

