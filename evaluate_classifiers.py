import numpy as np
import glob
import os
from pathlib import Path

for site_name in ["holstein_1_solar_farm"]:
  results = {}
  res_dir = Path("out") / site_name
  npy_files = glob.glob(str(res_dir / "*.npy"))
  date_indices = [int(os.path.basename(x).split("_")[0]) for x in npy_files]
  for date_index in range(1, max(date_indices)+1):
    gt = np.load(res_dir / "{}_gt.npy".format(date_index))[-1]
    mlp_pred = np.load(res_dir / "{}_mlp.npy".format(date_index))[-1]
    # rf_pred = np.load(res_dir / "{}_rf.npy".format(date_index))
    # gmm_pred = np.load(res_dir / "{}_gmm.npy".format(date_index))
    # breakpoint()
    results[date_index] = {
      "TP": np.sum((gt == 1) & (0 <= mlp_pred)),
      "FP": np.sum((gt == 0) & (0 <= mlp_pred)),
      "TN": np.sum((gt == 0) & (mlp_pred < 0)),
      "FN": np.sum((gt == 1) & (mlp_pred < 0)),
    }
    results[date_index]["IoU"] = results[date_index]["TP"] / (results[date_index]["TP"] + results[date_index]["FP"] + results[date_index]["FN"])
  print("="*80)
  print(site_name)
  print("IoU: {:.2f}".format(np.mean([x["IoU"] for x in results.values()])))
  print(results)



  
