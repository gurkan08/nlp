

from gensim.models.fasttext import FastText

corpus_dir = "fasttext_model/fasttext_corpus.txt"
model_dir = "fasttext_model/fasttext_model.bin"
embed_size = 100
epoch = 20

def fasttext_train_model():
    # read corpus
    with open(corpus_dir) as fp:
        corpus = fp.readlines()  # list of string

    model = FastText(size=embed_size, window=3, min_count=1)  # instantiate
    model.build_vocab(sentences=corpus)
    model.train(sentences=corpus, total_examples=len(corpus), epochs=epoch)  # train
    model.save(model_dir)  # save model

def infer_vector(text):
    # load model & infer word vector
    model = FastText.load(model_dir)
    vec = model.wv[text]
    return vec

#fasttext_train_model()
#vec = infer_vector("gürkan şahin")

