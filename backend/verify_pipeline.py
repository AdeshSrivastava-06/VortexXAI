import asyncio
import os
import sys

# Ensure backend directory in sys.path
backend_dir = os.path.dirname(__file__)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from main import (
    app, startup_event, predict, explain_point,
    get_forecast_10day, fetch_openmeteo_10day_forecast,
    PredictRequest, ExplainRequest, ensemble
)

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

    # 5. Test Open-Meteo 10-day future forecast service
    print("\n--- 5. Testing Open-Meteo 10-Day Future Forecast Service ---")
    om_res = await fetch_openmeteo_10day_forecast(lat=28.61, lon=77.23)
    print(f"Open-Meteo response source: {om_res['source']}, forecast days: {len(om_res['forecast'])}")
    assert len(om_res['forecast']) == 10, "Open-Meteo should return 10 days of future predictions!"
    day1 = om_res['forecast'][0]
    day10 = om_res['forecast'][9]
    print(f"Day +1: {day1['date']} | {day1['weather_desc']} | Temp: {day1['temp_min']} - {day1['temp_max']}°C | Rain: {day1['precipitation_mm']}mm | Bust Risk: {day1['bust_prob']*100:.1f}%")
    print(f"Day +10: {day10['date']} | {day10['weather_desc']} | Temp: {day10['temp_min']} - {day10['temp_max']}°C | Rain: {day10['precipitation_mm']}mm | Bust Risk: {day10['bust_prob']*100:.1f}%")

    # 6. Test /api/forecast_10day endpoint
    print("\n--- 6. Testing /api/forecast_10day Endpoint ---")
    ep_res = await get_forecast_10day(lat=19.07, lon=72.87)
    assert ep_res['status'] == 'success'
    assert len(ep_res['forecast']) == 10

    # 7. Test Explain Point Endpoint (SHAP + LIME + Open-Meteo integration)
    print("\n--- 7. Testing /api/explain_point Endpoint (SHAP + LIME + Open-Meteo) ---")
    req_exp = ExplainRequest(lat=15.5, lon=73.83, lead_day=3)
    res_exp = await explain_point(req_exp)
    print(f"Explain Point Response keys: {list(res_exp.keys())}")
    print(f"Bust Prob: {res_exp['bust_prob']}")
    print(f"Primary Driver: {res_exp['primary_driver']}")
    print(f"SHAP Values: {res_exp['shap_values']}")
    print(f"LIME Weights: {res_exp['lime_weights']}")
    print(f"LIME Agreement: {res_exp['lime_agreement']}")
    print(f"Dynamic Insight: {res_exp['dynamic_insight']}")
    print(f"Weather Source: {res_exp.get('weather_source')}")
    print(f"Open-Meteo 10-day items: {len(res_exp.get('openmeteo_10day', []))}")

    assert len(res_exp['shap_values']) == 5, "Should compute SHAP values for all 5 features!"
    assert len(res_exp['lime_weights']) == 5, "Should compute LIME weights for all 5 features!"
    assert len(res_exp['dynamic_insight']) > 10, "Should return dynamic operational insight!"
    assert len(res_exp.get('openmeteo_10day', [])) == 10, "Should include 10 days of Open-Meteo future data!"

    print("\n[SUCCESS] ALL VERIFICATION TESTS (INCLUDING OPEN-METEO 10-DAY FORECAST) PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    asyncio.run(main())
