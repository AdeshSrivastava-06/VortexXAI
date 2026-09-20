import os
import torch
import numpy as np
import lightgbm as lgb
import catboost as cb

backend_dir = os.path.dirname(__file__)

print("--- Checking Numpy artifacts ---")
coefs_path = os.path.join(backend_dir, "meta_learner_coefs.npy")
intercept_path = os.path.join(backend_dir, "meta_learner_intercept.npy")
if os.path.exists(coefs_path):
    coefs = np.load(coefs_path)
    print("meta_learner_coefs shape:", coefs.shape, "content:", coefs)
if os.path.exists(intercept_path):
    intercept = np.load(intercept_path)
    print("meta_learner_intercept shape:", intercept.shape, "content:", intercept)

print("\n--- Checking LightGBM model ---")
lgb_path = os.path.join(backend_dir, "lightgbm_bust_model.txt")
if os.path.exists(lgb_path):
    bst = lgb.Booster(model_file=lgb_path)
    print("LightGBM feature names:", bst.feature_name())
    print("LightGBM num features:", bst.num_feature())

print("\n--- Checking CatBoost model ---")
cb_path = os.path.join(backend_dir, "catboost_bust_model.cbm")
if os.path.exists(cb_path):
    cb_model = cb.CatBoostClassifier()
    cb_model.load_model(cb_path)
    print("CatBoost feature names:", cb_model.feature_names_)
    print("CatBoost num features:", len(cb_model.feature_names_) if cb_model.feature_names_ else "N/A")

print("\n--- Checking PyTorch ConvLSTM model ---")
pth_path = os.path.join(backend_dir, "convlstm_bust_model.pth")
if os.path.exists(pth_path):
    pth_obj = torch.load(pth_path, map_location='cpu', weights_only=False)
    print("PyTorch checkpoint type:", type(pth_obj))
    if isinstance(pth_obj, dict):
        print("Keys:", pth_obj.keys())
        for k, v in pth_obj.items():
            if hasattr(v, 'shape'):
                print(f"  {k}: tensor of shape {v.shape}")
            else:
                print(f"  {k}: {type(v)}")
    elif hasattr(pth_obj, 'state_dict'):
        print("Model class:", pth_obj.__class__.__name__)
        print(pth_obj)
    else:
        print("pth_obj:", pth_obj)

print("\n--- Checking ERA5 NetCDF file ---")
nc_path = os.path.join(backend_dir, "era5_india_latest.nc")
print("era5_india_latest.nc exists:", os.path.exists(nc_path))
