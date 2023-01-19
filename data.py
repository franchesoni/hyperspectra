# "Preparing hyperspectral data for annotation"
import os
import numpy as np

import tqdm

import utils
from config import data_path

site_names = sorted(os.listdir(data_path))



def get_data(site_name_index, date_index):
  # load all images in memory
  site_name = site_names[site_name_index]
  max_date_index = len(utils.get_filenames(os.path.join(data_path, site_name))[1])
  msis, rgbs, gts = [], [], []
  for date_index in tqdm.tqdm(range(min(max_date_index, date_index+1))):
    folder = os.path.join(data_path, site_name)
    msi, rgb, gt = utils.read_panel_image(folder, date_index)
    if gt is None:
      return
    msis += [msi]
    rgbs += [rgb]
    gts += [gt]

  msis = normalize_4d(msis)
  rgbs = normalize_4d(rgbs)
  gts = np.array(gts)

  X = msis.reshape(-1, msis.shape[-1])
  Y = gts.reshape(-1, gts.shape[-1])

  return msis, rgbs, gts, X, Y


def normalize_4d(msi):
  msi = np.array(msi)
  assert len(msi.shape) == 4
  mus = np.mean(msi, axis=(0, 1, 2))
  sigmas = np.std(msi, axis=(0, 1, 2))
  return (msi - mus) / sigmas



if __name__ == "__main__":
  site_name_index, date_index = 0, 3
  get_data(site_name_index, date_index)



  # out_dir = "out/{}".format(site_names[site_name_index])
  # Path(out_dir).mkdir(parents=True, exist_ok=True)

  # folder = os.path.join(data_path, site_names[site_name_index])
  # breakpoint()
  # msi, rgb, gt = utils.read_panel_image(folder, date_index)
  # msi3d, projunsup = mappings.apply_umap_unsup(msi)
  # msi3dsup, projsup = mappings.apply_umap_sup(msi, gt)
  # print('elapsed', time.time() -st)
  # utils.visualize(msi, rgb, gt, msi3d, msi3dsup)


  
# try histogram equalization