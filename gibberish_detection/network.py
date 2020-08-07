
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):

    # parameters
    hidden_dim = 50
    fasttext_embed_dim = 100 # update manually
    drop_rate = 0.2

    def __init__(self, num_class):
        super(Network, self).__init__()

        self.linear1 = nn.Linear(Network.fasttext_embed_dim, Network.hidden_dim)
        self.linear2 = nn.Linear(Network.hidden_dim, num_class)
        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(p=Network.drop_rate)

    def forward(self, batch):
        """
        :param batch: batch_first format
        :return: softmax output
        """
        out = self.relu(self.linear1(batch))
        out = self.dropout_layer(out)
        #print(out.size()) # torch.Size([25, 50])
        out = F.softmax(self.linear2(out), dim=1)
        #print(out.size()) # torch.Size([25, 2])
        return out

