from tinygrad import Tensor
from tinygrad import TinyJit
from tinygrad import nn
from tinygrad.nn.datasets import mnist
from tinygrad.helpers import GlobalCounters
from datetime import datetime

class Model():
  def __init__(self):
    self.layers = [lambda x: x.flatten(1), nn.Linear(784, 512), Tensor.relu, nn.Linear(512, 128), Tensor.relu, nn.Linear(128, 10)]
  def __call__(self, x: Tensor):
    return x.sequential(self.layers)


if __name__ == "__main__":
  xtrain, ytrain, xtest, ytest = mnist()
  m = Model()
  optim = nn.optim.Adam(nn.state.get_parameters(m))

  @TinyJit
  @Tensor.train()
  def train():
    optim.zero_grad()
    samples = Tensor.randint(500, high=xtrain.shape[0])
    loss = m(xtrain[samples]).sparse_categorical_crossentropy(ytrain[samples]).backward()
    optim.step()
    return loss
  
  for i in range(1):
    GlobalCounters.reset()
    loss = train()
    print(f"loss: {loss.item()}")