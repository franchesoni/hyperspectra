import os
import glob

import numpy as np
import skimage
import iio
import matplotlib.pyplot as plt


bandes = [
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        "B10",
        "B11",
        "B12",
    ]



def get_filenames(folder):
  # search for spectral files
  series = {}
  for b in bandes:
    series[b] = []
    for x in os.walk(folder):
        for y in sorted(glob.glob(os.path.join(x[0], "*" + b + ".tif"))):
            series[b] += [y]
  # search for ground truth masks
  gt_series = None
  for x in os.walk(folder):
    if "manual_segmentation_masks" in x[0]:
      gt_series = sorted(glob.glob(os.path.join(x[0], "*")))

  # search for the RGB series
  rgb_series = {}
  for b in ["B02", "B03", "B04"]:
    rgb_series[b] = []
    for x in os.walk(folder):
      for y in sorted(glob.glob(os.path.join(x[0], "*"+b+".tif"))):
        rgb_series[b] += [y]

  # check bands
  for band, filelist in series.items():
    if len(filelist) == 0:
      raise RuntimeError("Missing band %s" % (band))
  
  # check rgb bands
  for band, filelist in rgb_series.items():
    if len(filelist) == 0:
      raise RuntimeError("Missing band %s" % (band))

  return series, gt_series, rgb_series

def read_panel_image(folder, date_index, normalize=False):
    series, gt_series, rgb_series = get_filenames(folder)
    date_index = np.clip(0, len(series["B02"]) - 1, date_index)
    rgb_image = np.stack(
        [
            iio.read(rgb_series["B04"][date_index])[..., 0],
            iio.read(rgb_series["B03"][date_index])[..., 0],
            iio.read(rgb_series["B02"][date_index])[..., 0],
        ],
        axis=2,
    )

    factors = {
        "B01": 6,
        "B02": 1,
        "B03": 1,
        "B04": 1,
        "B05": 2,
        "B06": 2,
        "B07": 2,
        "B08": 1,
        "B8A": 2,
        "B09": 6,
        "B10": 6,
        "B11": 2,
        "B12": 2,
    }
    multispectral_image = np.stack(
        [
            skimage.transform.rescale(
                iio.read(series[b][date_index])[..., 0],
                scale=factors[b],
                order=5,
            )
            for b in series
        ],
        axis=2,
    )
    ground_truth = make_single_channel(binarize(iio.read(gt_series[date_index]))) if gt_series is not None else None
    if normalize:
        return (norm_fn(multispectral_image)*255).astype(np.uint8), (norm_fn(rgb_image)*255).astype(np.uint8)
    return multispectral_image, rgb_image, ground_truth

def norm_fn(x):
    return (x - x.min()) / (x.max() - x.min())

def binarize(x):
    return 1 * (x > x.max() / 2)

def make_single_channel(x):
    assert (np.unique(x) == [0, 1]).all()
    return np.any(x, axis=2, keepdims=True) * 1

def make_showable(img):
    return img[:, :, :3]

def visualize(*images):
    n_imgs = len(images)
    side = int(np.ceil(np.sqrt(n_imgs)))
    # create grid of images
    plt.figure()
    for i, img in enumerate(images):
        plt.subplot(side, side, i + 1)
        plt.imshow(norm_fn(make_showable(img)))
        plt.axis("off")
    plt.show()

def save_arrays(arrays: list, names: list):
    assert len(arrays) == len(names)
    for name, arr in zip(names, arrays):
        np.save(f"out/{name}.npy", arr)

def preprocess(*images):
    # here we could do, for instance, histogram equalization
    s = 1000
    return [img[:s, :s] for img in images]