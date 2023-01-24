import random
import torch
import numpy as np
import tqdm
import sklearn.ensemble
import jax.random
from functools import partial

from data import get_data, get_max_date_index, site_names
import gmm_jax
from utils import save_arrays



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

  return torch.nn.Sequential(mlp, torch.nn.Sigmoid()).eval()
###############

########## Gaussian #########
def build_and_train_gaussian(X, Y):
  X_pos = X[(Y == 1).squeeze()]
  means, invcov = gmm_jax.gaussian(X_pos)
  gmm_fn = partial(gmm_jax.gaussian_predict, mus=means, invsigma=invcov)
  return gmm_fn
###############

########## KNN #########
def build_and_train_rf(X, Y):
  rf_cls = sklearn.ensemble.RandomForestClassifier(class_weight="balanced", n_jobs=-1, verbose=1)
  rf_cls = rf_cls.fit(X, Y.squeeze())
  def rf_fn(x):
    return rf_cls.predict_proba(x)[:, 1]
  return rf_fn
###############


def predict(msi, predictor_fn, torchtensor):
  if torchtensor:
    with torch.no_grad():
      new_msi = predictor_fn(torch.from_numpy(msi.reshape(-1, msi.shape[-1])).float()).numpy().reshape(msi.shape[:-1] + (1,))
  else:
      new_msi = predictor_fn(msi.reshape(-1, msi.shape[-1])).reshape(msi.shape[:-1] + (1,))
  return new_msi





if __name__ == '__main__':
  torch.manual_seed(0)
  random.seed(0)
  np.random.seed(0)
  
  # # generate samples from a bidimiensional gaussian distribution
  # mus = np.array([[-2, 1]])
  # sigma = np.array([[1, 0.5], [0.5, 1]])
  # X = np.random.multivariate_normal(mus[0], sigma, size=1000)
  # import matplotlib.pyplot as plt 
  # plt.scatter(X[:, 0], X[:, 1])
  # plt.show()

  # Y = np.ones((X.shape[0], 1))
  # gmm_fn = build_and_train_gaussian(X, Y)
  # pred_gmm = np.exp(np.array(predict(X, gmm_fn, torchtensor=False)))

  # ax = plt.figure().add_subplot(projection='3d')
  # ax.plot_trisurf(X[:, 0], X[:, 1], pred_gmm.squeeze(), antialiased=True)
  # plt.show()

  for site_name_index in [4, 5]:
    site_name = site_names[site_name_index]
    max_date_index = get_max_date_index(site_name_index)
    for date_index in range(max_date_index):
      msi, rgb, gt, X, Y = get_data(site_name_index, date_index)
      if msi.shape[0] == 1:
        save_arrays([rgb, gt], [f"{site_name}/0_rgb", f"{site_name}/0_gt"])
        continue
        
      mlp = build_and_train_mlp(X[:-1], Y[:-1], 500)
      pred_mlp = predict(msi, mlp, torchtensor=True)

      gmm_fn = build_and_train_gaussian(X[:-1], Y[:-1])
      pred_gmm = predict(msi, gmm_fn, torchtensor=False)

      rf_fn = build_and_train_rf(X[:-1], Y[:-1])
      pred_rf = predict(msi, rf_fn, torchtensor=False)

      save_arrays([rgb, gt, pred_mlp, 
     pred_gmm, pred_rf
      ], 
      [f"{site_name}/{date_index}_rgb", f"{site_name}/{date_index}_gt", f"{site_name}/{date_index}_mlp",
      f"{site_name}/{date_index}_gmm", f"{site_name}/{date_index}_rf"
      ])


  

