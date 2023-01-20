import torch
import numpy as np
import tqdm
import skimage
import sklean.decomposition

from data import get_data

class MyPCA(torch.nn.Module):
  """A class to train an autoencoder that has one of the embeddings given by an external mlp"""
  def __init__(self, input_size, trained_mlp, hidden_size=2):
    super(MyPCA, self).__init__()
    self.trained_mlp = trained_mlp
    self.lin1 = torch.nn.Linear(input_size, hidden_size)
    self.decoder = torch.nn.Linear(hidden_size + 1, input_size)

  def encode(self, x):
    with torch.no_grad():
      x0 = self.trained_mlp(x)
    x1 = self.lin1(x)
    # concatenate to have (B, hidden_size + 1)
    return torch.cat([x0, x1], dim=-1) 
    
  def forward(self, x):
    return self.decoder(self.encode(x))
  
  

def build_mlp(input_size, n_layers=2, hidden_size=16):
  layers = []
  for i in range(n_layers):
    layers.append(torch.nn.Linear(input_size, hidden_size))
    layers.append(torch.nn.GELU())
    input_size = hidden_size
  layers.append(torch.nn.Linear(input_size, 1))
  return torch.nn.Sequential(*layers)

def train_mapper(X, Y):
  X_pos, X_neg = X[(Y==1).squeeze()], X[(Y==0).squeeze()]
  diffs = X_pos[:, None, :] - X_neg[None, :, :]  # (n_pos, n_neg, n_features)
  pca = sklearn.decomposition.PCA(n_components=3, random_state=0)
  print("Fitting diff PCA")
  pca = pca.fit(diffs.reshape(-1, diffs.shape[-1]))

  return pca.predict


def to_01(img):
  # normalize each channel individually to [0,1]
  if len(img.shape) == 3:
    axis = (0, 1)
  elif len(img.shape) == 4:
    axis = (0, 1, 2)
  else:
    raise ValueError("img must be 3 or 4 dimensional")
  img = (img - img.min(axis=axis, keepdims=True)) / (img.max(axis=axis, keepdims=True) - img.min(axis=axis, keepdims=True))
  return img


def map_colors(msi, pca_predict):
  new_msi = pca_predict(msi.reshape(-1, msi.shape[-1])).reshape(msi.shape[:-1] + (3,))
  return to_01(new_msi)



if __name__ == '__main__':
  site_name_index, date_index = 0, 10
  msi, rgb, gt, X, Y = get_data(site_name_index, date_index)
  pca_predict_fn = train_mapper(X[:-1], Y[:-1])
  frgb = map_colors(msi, pca_predict_fn)
  import matplotlib.pyplot as plt
  for i in range(frgb.shape[0]):
    plt.imsave(f'out/frgb_{i}.jpg', frgb[i])

  print('fin')
  

