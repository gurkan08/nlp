
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
import os

def download_and_save_bert_model(bert_model_name, model_save_dir):
    bert_config = AutoConfig.from_pretrained(bert_model_name)  # config
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    bert_model = AutoModel.from_pretrained(bert_model_name)
    print("downloaded bert model config: ", bert_model.config)
    # save downloaded tokenizer / models
    bert_tokenizer.save_pretrained(model_save_dir)
    bert_model.save_pretrained(model_save_dir)  # aynı dir içine kaydet

def load_bert_model(bert_hidden_size):
    bert_config = AutoConfig.from_pretrained("model/", hidden_size=bert_hidden_size)
    bert_tokenizer = AutoTokenizer.from_pretrained('model/')
    bert_model = AutoModel.from_config(bert_config) # fom_pretrained değil from_config olacakmış :)
    print("loaded bert model config: ", bert_model.config)  # ok sonunda :)

    for param in bert_model.parameters():  # freeze bert model params (not bert training)
        param.requires_grad = False

    return bert_tokenizer, bert_model

class Params(object):

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

    bert_model_name = "dbmdz/bert-base-turkish-cased"
    download_or_load_bert_model = "load" # download | load
    bert_tokenizer = None
    bert_model = None
    batch_size = 128 # çok yapma patliyi : ))))
    epoch = 10
    lr = 0.0025
    hidden_1_size = 128
    hidden_2_size = 64
    bert_sequence_max_length = 50
    bert_hidden_size = 384 #384 # (num_attention_heads=12 bunun katı olmalıymış :))))) # default: 768
    batch_shuffle = True

    model = None
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
    if download_or_load_bert_model == "download":
        download_and_save_bert_model(bert_model_name, model_dir)
    elif download_or_load_bert_model == "load":
        bert_tokenizer, bert_model = load_bert_model(bert_hidden_size)

