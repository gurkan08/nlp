
from gensim.models.fasttext import FastText
from fasttext_train import model_dir

# load pretrained fasttext model
fasttext_model = FastText.load(model_dir)

class MyDataloader(object):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # convert string type X[index] to fasttext vector numeric type
        fasttext_vec = fasttext_model.wv[self.X[index]]
        return fasttext_vec, self.y[index]

