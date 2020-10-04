
"""
# direct keras (keras API) modules
from keras.layers import Input, Embedding, LSTM, Dense, Dropout, SimpleRNN
from keras.models import Model
"""

# tf==1.15.0 modules
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

from lstm_model.params import Params

class ClassificationModel(object):

    def __init__(self,
                 max_sentence_size,
                 embed_size,
                 vocab_size,
                 lstm_units,
                 dense_size,
                 label_size):

        # mask_zero=True zero_padding, trainable=False fasttext embedding init
        input_layer = Input(shape=(max_sentence_size,))

        embed_layer = Embedding(input_dim=vocab_size,
                                output_dim=embed_size,
                                mask_zero=True,
                                #weights=[Params.embedding_matrix],
                                trainable=True)(input_layer)

        lstm_layer = LSTM(lstm_units,
                          return_sequences=False,
                          return_state=False,
                          trainable=True)(embed_layer)
        # print(lstm_layer.shape)

        dense_layer = Dense(dense_size,
                            activation="relu",
                            trainable=True)(lstm_layer)
        # print(dense_layer.shape)

        drop_layer = Dropout(rate=Params.drop_out)(dense_layer)

        output_layer = Dense(label_size,
                             activation="softmax",
                             trainable=True)(drop_layer)
        # print(output_layer.shape)

        self.model = Model(input_layer, output_layer)

    def get_model(self):
        return self.model