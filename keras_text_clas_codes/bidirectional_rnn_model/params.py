
import fasttext
import os

class Params(object):

    excel_dir = "data/data_6_class_balanced.xlsx"

    preprocess_steps = {
        "lowercase": True,
        "remove_punctuations": True,
        "remove_numbers": True,
        "remove_stop_words": True,
        "zemberek_stemming": False,
        "first_5_char_stemming": True,
        "data_shuffle": True
    }

    max_sent_size = None # mean sentence size in train dataset
    embedding_matrix = None
    embed_size = 300 # 300 for fasttext embeds
    rnn_units = 100
    dense_size = 50
    drop_out = 0.3

    epochs = 50
    optimizer = "adam" # adam|sgd
    batch_size = 1024 # 64
    lr = 0.025 # 5e-5 # 0.00025
    early_stop_patience = 100
    ReduceLROnPlateau_factor = 0.9 # 0.1 çok fazla azaltıyor ..! # new_lr = lr * factor
    ReduceLROnPlateau_patience = 2
    ReduceLROnPlateau_min_lr = 1e-6

    test_size = 0.3
    validation_split = 0.1
    shuffle_count = 50

    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    model_dir = "model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    plot_dir = "plot"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    #fasttext_model = fasttext.load_model("model/cc.tr.300.bin")
