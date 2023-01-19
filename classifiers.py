import torch
import numpy as np
import tqdm
import skimage
import sklearn
import jax.random
from functools import partial

from data import get_data
import gmm_jax


############# MLP ############
def build_mlp(input_size, n_layers=2, hidden_size=16):
  layers = []
  for i in range(n_layers):
    layers.append(torch.nn.Linear(input_size, hidden_size))
    layers.append(torch.nn.GELU())
    input_size = hidden_size
  layers.append(torch.nn.Linear(input_size, 1))
  return torch.nn.Sequential(*layers)

def build_and_train_mlp(X, Y, n_epochs=1000):
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

  return mlp
###############

########## Gaussian #########
def build_and_train_gaussian(X, Y):
  X_pos = X[(Y == 1).squeeze()]
  means, covs, weights = gmm_jax.gmm(jax.random.PRNGKey(0), X_pos, 1)
  gmm_fn = partial(gmm_jax.predict, means=means, covs=covs, weights=weights)
  return gmm_fn





def predict(msi, predictor_fn, torchtensor):
  if torchtensor:
    with torch.no_grad():
      new_msi = predictor_fn(torch.from_numpy(msi.reshape(-1, msi.shape[-1])).float()).numpy().reshape(msi.shape[:-1] + (1,))
  else:
      new_msi = predictor_fn(msi.reshape(-1, msi.shape[-1])).reshape(msi.shape[:-1] + (1,))
  return new_msi





if __name__ == '__main__':
  site_name_index, date_index = 0, 10
  msi, rgb, gt, X, Y = get_data(site_name_index, date_index)

  # mlp = build_and_train_mlp(X[:-1], Y[:-1], 10)#00)
  # pred_mlp = predict(msi, mlp, torchtensor=True)

  gmm_fn = build_and_train_gaussian(X[:-1], Y[:-1])
  pred_gmm = predict(msi, gmm_fn, torchtensor=False)

  breakpoint()

  

