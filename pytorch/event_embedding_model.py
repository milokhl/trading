# Model from https://www.ijcai.org/Proceedings/15/Papers/329.pdf
import torch.nn as nn
from torch import cat

class EventEmbedder(nn.Module):
  def __init__(self, word_dim, relation_dim, event_dim):
    """
    word_dim: the dimension of a word embedding
    relation_dim: the intermediate dimension of a relationship
    event_dim: the desired dimension of the event embedding
    """
    super(EventEmbedder, self).__init__()

    self.R1 = BilinearTensorLayer(word_dim, relation_dim)
    self.R2 = BilinearTensorLayer(word_dim, relation_dim)
    self.EmbedLayer = BilinearTensorLayer(relation_dim, event_dim)
    self.ScoreLayer = nn.Linear(event_dim, 1)
    self.tanh = nn.Tanh()
    print('Initialized EventEmbedder model!')

  def forward(self, O1, T, O2):
    R1 = self.tanh(self.R1(O1, T))
    R2 = self.tanh(self.R2(O2, T))
    embedding = self.tanh(self.EmbedLayer(R1, R2))
    score = self.ScoreLayer(embedding)
    return score, embedding


class BilinearTensorLayer(nn.Module):
  def __init__(self, input_vec_dim, output_vec_dim):
    super(BilinearTensorLayer, self).__init__()
    self.input_vec_dim = input_vec_dim
    self.output_vec_dim = output_vec_dim

    # Set up tensor, linear, and bias weights.
    self.bilinear = nn.Bilinear(input_vec_dim, input_vec_dim, output_vec_dim, bias=False)
    self.linear = nn.Linear(2 * input_vec_dim, output_vec_dim, bias=True)

  def forward(self, vec1, vec2):
    return self.bilinear(vec1, vec2) + self.linear(cat((vec1, vec2), 1))