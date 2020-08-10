
import torch

class MyDataloader(object):

    def __init__(self, X, y, word2id, tag2id, max_sentence_len, empty_tag):
        self.X = X
        self.y = y
        self.word2id = word2id
        self.tag2id = tag2id
        self.max_sentence_len = max_sentence_len
        self.empty_tag = empty_tag

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):

        """
        for X data
        """
        # convert string type self.X[index] to numeric type (word2id) format
        X_numeric = []
        for word in self.X[index].split():
            X_numeric.append(self.word2id[word.strip()])

        # zero padding manually, because torch.nn.utils.rnn.pad_sequence() doesn't run correctly ??? (look later)
        for i in range(len(X_numeric), self.max_sentence_len):
            X_numeric.append(0)  # zero pad to back

        # convert to tensor, because now all batch elements is same size :)
        X_numeric = torch.tensor(X_numeric)

        """
        for y data
        """
        # convert string type self.y[index] to numeric type (tag2id) format
        y_numeric = []
        for tag in self.y[index].split():
            y_numeric.append(self.tag2id[tag.strip()])

        # pad manually using <o> -self.empty_tag- tag :)
        for i in range(len(y_numeric), self.max_sentence_len):
            y_numeric.append(self.tag2id[self.empty_tag])

        # convert to tensor, because now all batch elements is same size :)
        y_numeric = torch.tensor(y_numeric)

        return X_numeric, y_numeric, len(self.X[index].split())


