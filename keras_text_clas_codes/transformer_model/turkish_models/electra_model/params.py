
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
import os

def download_and_save_model(model_name, model_save_dir):
    config = AutoConfig.from_pretrained(model_name)  # config
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    print("downloaded model config: ", model.config)
    # save downloaded tokenizer / models
    tokenizer.save_pretrained(model_save_dir)
    model.save_pretrained(model_save_dir)  # aynı dir içine kaydet

def load_model(model_hidden_size):
    config = AutoConfig.from_pretrained("model/", hidden_size=model_hidden_size)
    tokenizer = AutoTokenizer.from_pretrained('model/')
    model = AutoModel.from_config(config) # fom_pretrained değil from_config olacakmış :)
    print("loaded model config: ", model.config)  # ok sonunda :)

    for param in model.parameters():  # freeze model params
        param.requires_grad = False

    return tokenizer, model

class Params(object):

    """
    electra model
    """

    dataset_dir = "data/4900_news.xlsx"

    preprocess_steps = {
        "lowercase": True,
        "remove_punctuations": True,
        "remove_numbers": True,
        "remove_stop_words": True,
        "zemberek_stemming": False,
        "first_5_char_stemming": False,
        "data_shuffle": True
    }

    test_split_rate = 0.3
    shuffle_count = 5
    label_tokenizer_name = "label_tokenizer.pickle"
    label_tokenizer = None

    model_name = "erayyildiz/electra-turkish-cased"
    download_or_load_model = "load" # download | load
    tokenizer = None
    model = None
    batch_size = 128 # çok yapma patliyi : ))))
    epoch = 10
    lr = 0.0025
    hidden_1_size = 128
    hidden_2_size = 64
    sequence_max_length = 50
    model_hidden_size = 384 #384 # "dim": 768
    batch_shuffle = True

    pipeline_model = None
    optimizer = None
    criterion = None

    use_cuda = torch.cuda.is_available()
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    plot_dir = "plot"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    model_dir = "model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # download | load bert model
    if download_or_load_model == "download":
        download_and_save_model(model_name, model_dir)
    elif download_or_load_model == "load":
        tokenizer, model = load_model(model_hidden_size)

