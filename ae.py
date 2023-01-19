import torch
import numpy as np
import tqdm
import skimage

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

def train_mapper(X, Y, n_epochs=1000):
  X, Y = torch.from_numpy(X).float(), torch.from_numpy(Y).float()
  pos_weight = (Y.shape[0] - Y.sum()) / Y.sum()

  print("Training MLP classifier")
  mlp = build_mlp(X.shape[-1])
  optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)
  pbar = tqdm.tqdm(range(n_epochs))
  for epoch_ind in pbar:
    optimizer.zero_grad()
    y_pred = mlp(X)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(y_pred, Y, pos_weight=pos_weight)
    loss.backward()
    optimizer.step()
    pbar.set_description(f"loss = {loss.item()}")
  acc = (1.*(torch.nn.functional.sigmoid(y_pred).round() == Y)).mean()
  print(f"Accuracy = {acc}")

  print("Training autoencoder")
  autoencoder = MyPCA(X.shape[-1], mlp)
  optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
  pbar = tqdm.tqdm(range(n_epochs))
  for epoch_ind in pbar:
    optimizer.zero_grad()
    y_pred = autoencoder(X)
    loss = torch.nn.functional.mse_loss(y_pred, X)
    loss.backward()
    optimizer.step()
    pbar.set_description(f"loss = {loss.item()}")

  return autoencoder

labmins = np.array([   0.        ,  -86.18302974, -107.85730021])
labmaxs = np.array([100.        ,  98.23305386,  94.47812228])


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

def colorspace_handling(img):
  img = to_01(img)
  # convert to lab range
  img = img * (labmaxs - labmins) + labmins
  # convert to rgb
  return skimage.color.lab2rgb(img)

def map_colors(msi, autoencoder):
  with torch.no_grad():
    new_msi = autoencoder.encode(torch.from_numpy(msi.reshape(-1, msi.shape[-1])).float()).numpy().reshape(msi.shape[:-1] + (3,))
  return colorspace_handling(new_msi)



if __name__ == '__main__':
  site_name_index, date_index = 0, 10
  msi, rgb, gt, X, Y = get_data(site_name_index, date_index)
  autoencoder = train_mapper(X[:-1], Y[:-1], 1000)
  frgb = map_colors(msi, autoencoder)
  import matplotlib.pyplot as plt
  for i in range(frgb.shape[0]):
    plt.imsave(f'out/frgb_{i}.jpg', frgb[i])
  breakpoint()

  print('fin')
  

