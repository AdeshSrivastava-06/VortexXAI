import asyncio
import os
import sys

# Ensure backend directory in sys.path
backend_dir = os.path.dirname(__file__)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from main import app, startup_event, predict, explain_point, PredictRequest, ExplainRequest, ensemble

async def main():
    print("=== Running VortexXAI End-to-End ML Pipeline Verification ===")
    
    # 1. Trigger startup sequence
    print("\n--- 1. Triggering FastAPI Startup Sequence ---")
    await startup_event()
    
    # 2. Check ensemble state
    print("\n--- 2. Verifying Model Artifact Initialization ---")
    print(f"Ensemble Ready: {ensemble.ready}")
    print(f"PyTorch ConvLSTM Model loaded: {ensemble.convlstm_model is not None}")
    print(f"LightGBM Booster loaded: {ensemble.lgb_booster is not None}")
    print(f"CatBoost Model loaded: {ensemble.cat_model is not None}")
    print(f"Meta-Learner Coefs loaded: {ensemble.meta_coefs is not None}")
    print(f"NetCDF Dataset loaded: {ensemble.netcdf_ds is not None}")
    
    assert ensemble.convlstm_model is not None, "ConvLSTM model should be loaded!"
    assert ensemble.lgb_booster is not None, "LightGBM booster should be loaded!"
    assert ensemble.cat_model is not None, "CatBoost model should be loaded!"
    assert ensemble.meta_coefs is not None, "Meta-learner coefs should be loaded!"

    # 3. Test NetCDF spatial crop extraction
    print("\n--- 3. Testing Spatial Tensor Slicing ---")
    crop = ensemble.extract_spatial_crop(lat=15.5, lon=73.83, lead_day=3)
    print(f"Spatial crop tensor shape: {crop.shape}, dtype: {crop.dtype}")
    assert crop.ndim == 5, "Spatial crop should be a 5D tensor (1, Time, Channels, Lat, Lon)!"
    assert crop.shape[2] == 4, "Spatial crop should have 4 physical weather channels!"

    # 4. Test Predict Endpoint
    print("\n--- 4. Testing /api/predict Endpoint ---")
    req_pred = PredictRequest(lead_day=3)
    res_pred = await predict(req_pred)
    print(f"Predict response status: {res_pred['status']}, count: {res_pred['count']}")
    assert res_pred['status'] == 'success'
    assert len(res_pred['data']) > 0
    sample_pt = res_pred['data'][0]
    print(f"Sample Point: ID={sample_pt.id}, State={sample_pt.state_name}, Risk={sample_pt.bust_prob}, Driver={sample_pt.primary_driver}")

    # 5. Test Explain Point Endpoint
    print("\n--- 5. Testing /api/explain_point Endpoint (SHAP + LIME) ---")
    req_exp = ExplainRequest(lat=15.5, lon=73.83, lead_day=3)
    res_exp = await explain_point(req_exp)
    print(f"Explain Point Response keys: {list(res_exp.keys())}")
    print(f"Bust Prob: {res_exp['bust_prob']}")
    print(f"Primary Driver: {res_exp['primary_driver']}")
    print(f"SHAP Values: {res_exp['shap_values']}")
    print(f"LIME Weights: {res_exp['lime_weights']}")
    print(f"LIME Agreement: {res_exp['lime_agreement']}")
    print(f"Dynamic Insight: {res_exp['dynamic_insight']}")

    assert len(res_exp['shap_values']) == 5, "Should compute SHAP values for all 5 features!"
    assert len(res_exp['lime_weights']) == 5, "Should compute LIME weights for all 5 features!"
    assert len(res_exp['dynamic_insight']) > 10, "Should return dynamic operational insight!"

    print("\n[SUCCESS] ALL VERIFICATION TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    asyncio.run(main())
