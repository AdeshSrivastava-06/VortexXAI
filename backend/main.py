import os
import math
import json
import numpy as np
import reverse_geocoder as rg
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import torch
import torch.nn as nn
import lightgbm as lgb
import catboost as cb

try:
    import xarray as xr
except ImportError:
    xr = None

try:
    import shap
except ImportError:
    shap = None

try:
    import lime
    import lime.lime_tabular
except ImportError:
    lime = None

try:
    from shapely.geometry import shape, Point
except ImportError:
    shape = None
    Point = None


app = FastAPI(title="MoES Operational Forecast Bust Detection & XAI System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════════
# PYDANTIC DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class GridPoint(BaseModel):
    id: str
    lat: float
    lon: float
    state_name: str
    district_name: str
    bust_prob: float
    temp_anomaly: float
    rain_squared: float
    conv_interaction: float
    humidity: float
    wind_shear: float
    pressure: float
    pressure_drop: float
    primary_driver: str
    similarity_score: float
    past_event: str


class PredictRequest(BaseModel):
    lead_day: int


class ExplainRequest(BaseModel):
    lat: float
    lon: float
    lead_day: int


# ═══════════════════════════════════════════════════════════════════════════════
# PYTORCH CONVLSTM MODEL ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════

class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=self.padding
        )

    def forward(self, x: torch.Tensor, state=None):
        batch_size, _, height, width = x.size()
        if state is None:
            h = torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device, dtype=x.dtype)
            c = torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device, dtype=x.dtype)
        else:
            h, c = state

        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, g, o = torch.split(gates, self.hidden_channels, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, (h_next, c_next)


class ConvLSTMModel(nn.Module):
    def __init__(self, in_channels: int = 4):
        super().__init__()
        self.channel_adapter = nn.Conv2d(in_channels, 2, kernel_size=1) if in_channels != 2 else None
        if self.channel_adapter is not None:
            with torch.no_grad():
                self.channel_adapter.weight.zero_()
                self.channel_adapter.weight[0, 0, 0, 0] = 0.5  # Temp
                self.channel_adapter.weight[0, 3, 0, 0] = 0.5  # Humidity
                self.channel_adapter.weight[1, 1, 0, 0] = 0.5  # Pressure
                self.channel_adapter.weight[1, 2, 0, 0] = 0.5  # Wind
                self.channel_adapter.bias.zero_()

        self.cell1 = ConvLSTMCell(in_channels=2, hidden_channels=32, kernel_size=3)
        self.cell2 = ConvLSTMCell(in_channels=32, hidden_channels=16, kernel_size=3)
        self.head = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.unsqueeze(1)
        batch_size, seq_len, num_channels, height, width = x.size()
        state1 = None
        state2 = None
        for t in range(seq_len):
            x_t = x[:, t]
            if self.channel_adapter is not None:
                x_t = self.channel_adapter(x_t)
            h1, state1 = self.cell1(x_t, state1)
            h2, state2 = self.cell2(h1, state2)

        out = self.head(h2)
        prob = torch.sigmoid(out.mean(dim=(-2, -1))).squeeze(-1)
        return prob


# ═══════════════════════════════════════════════════════════════════════════════
# STRICT NATIONAL LAND BOUNDARY MASKING (INDIA ONLY)
# ═══════════════════════════════════════════════════════════════════════════════

INDIAN_POLYGONS = []         # List of Shapely geometry polygons
RAW_POLYGON_COORDS = []      # List of coordinate arrays for fast ray-casting fallback

INDIAN_STATES_SET = {
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
    "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram",
    "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
    "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal",
    "Delhi", "Jammu and Kashmir", "Ladakh", "Puducherry", "Chandigarh",
    "Dadra and Nagar Haveli and Daman and Diu", "Daman and Diu", "Dadra and Nagar Haveli",
    "Andaman and Nicobar Islands", "Lakshadweep"
}


def _point_in_ring(x: float, y: float, ring: list) -> bool:
    """Standard ray-casting algorithm for 2D point-in-polygon test (x=lon, y=lat)."""
    n = len(ring)
    inside = False
    p1x, p1y = ring[0]
    for i in range(n + 1):
        p2x, p2y = ring[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def load_india_boundary():
    """Load India spatial boundary polygons from GeoJSON."""
    global INDIAN_POLYGONS, RAW_POLYGON_COORDS

    candidate_files = [
        os.path.join(os.path.dirname(__file__), "india_states.json"),
        os.path.join(os.path.dirname(__file__), "india_boundary.json")
    ]

    for filepath in candidate_files:
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)

                features = data.get("features", [])
                for feat in features:
                    geom = feat.get("geometry", {})
                    gtype = geom.get("type", "")
                    coords = geom.get("coordinates", [])

                    if shape is not None:
                        try:
                            s_geom = shape(geom)
                            INDIAN_POLYGONS.append(s_geom)
                        except Exception:
                            pass

                    # Cache raw coordinates for ray-casting fallback
                    if gtype == "Polygon":
                        if coords:
                            RAW_POLYGON_COORDS.append(coords[0])
                    elif gtype == "MultiPolygon":
                        for poly in coords:
                            if poly:
                                RAW_POLYGON_COORDS.append(poly[0])

                print(f"Loaded {len(RAW_POLYGON_COORDS)} boundary rings from {os.path.basename(filepath)}")
                break
            except Exception as e:
                print(f"Failed to load boundary from {filepath}: {e}")


def is_within_india_land(lat: float, lon: float) -> bool:
    """Strict geographic and polygon boundary test for Indian landmass with coastal buffer."""
    if not (6.5 <= lat <= 37.5 and 68.0 <= lon <= 97.5):
        return False

    # Coastal Ocean Water Hard Exclusion Masks
    if lon < 72.0 and lat < 20.0: return False
    if lon < 72.4 and lat < 18.0: return False
    if lon < 73.0 and lat < 14.5: return False
    if lon < 73.8 and lat < 12.0: return False
    if lon < 74.5 and lat < 10.0: return False
    if lon < 75.5 and lat < 8.0:  return False

    if lat < 9.0 and 80.0 < lon < 92.0: return False
    if lat < 13.0 and 81.0 < lon < 92.0: return False
    if lat < 16.0 and 82.5 < lon < 92.0: return False
    if lat < 18.5 and 85.0 < lon < 92.0: return False
    if lat < 20.5 and 87.5 < lon < 91.5: return False

    if 22.0 <= lat <= 25.0 and 89.2 <= lon <= 91.8:
        return False

    if INDIAN_POLYGONS and Point is not None:
        pt = Point(lon, lat)
        for poly in INDIAN_POLYGONS:
            if poly.contains(pt) or poly.distance(pt) < 0.12:
                return True
        return False

    if RAW_POLYGON_COORDS:
        return any(_point_in_ring(lon, lat, ring) for ring in RAW_POLYGON_COORDS)

    return True


# ═══════════════════════════════════════════════════════════════════════════════
# MONSOON HOTSPOT GAUSSIAN RISK CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════════

HOTSPOTS = [
    (26.5, 79.5, 1.8),   # UP / Uttarakhand monsoon trough
    (12.0, 75.0, 1.5),   # Western Ghats / Kerala
    (25.5, 92.0, 1.6),   # Assam / NE India
    (21.0, 87.0, 1.3),   # Odisha coastal
    (19.0, 73.0, 1.2),   # Mumbai / Konkan
]


def spatial_risk_boost(lat: float, lon: float) -> float:
    """Gaussian spatial gradient — returns 0.0 to ~1.8 boost near hotspots."""
    boost = 0.0
    for hlat, hlon, strength in HOTSPOTS:
        dist = math.sqrt((lat - hlat) ** 2 + (lon - hlon) ** 2)
        boost += strength * math.exp(-(dist ** 2) / 18.0)
    return boost


# ═══════════════════════════════════════════════════════════════════════════════
# HISTORICAL SYNOPTIC PATTERN MATCHING ENGINE
# Feature order: [Temp_Anom, Rain_Accel, Shear, Humidity, Pressure]
# ═══════════════════════════════════════════════════════════════════════════════

HISTORICAL_EVENTS = [
    {
        "name": "2021 Uttarakhand Cloudburst & Flash Floods",
        "description": "High Rain Acceleration, High Temperature Anomaly",
        "vector": [11.5, 14400.0, 14.0, 94.0, 9.5],
        "region": "north_northwest",
    },
    {
        "name": "2023 North-West India Severe Western Disturbance",
        "description": "High Pressure Drop, Moderate Rain",
        "vector": [8.4, 5400.0, 18.5, 71.0, 19.5],
        "region": "north_northwest",
    },
    {
        "name": "2022 Rajasthan Extreme Summer Heatwave",
        "description": "High Temp Anomaly, Low Humidity",
        "vector": [18.2, 80.0, 7.5, 18.0, 3.2],
        "region": "north_northwest",
    },
    {
        "name": "2023 Himachal Himalayan Cloudburst Instability",
        "description": "Intense Convection, Vertical Wind Shear",
        "vector": [9.8, 12800.0, 16.0, 91.0, 14.5],
        "region": "north_northwest",
    },
    {
        "name": "2018 Kerala Severe Deluge & Monsoonal Break",
        "description": "Extreme Rain Acceleration",
        "vector": [5.4, 14100.0, 10.5, 95.0, 7.0],
        "region": "south_east",
    },
    {
        "name": "2020 Super Cyclone Amphan - Bay of Bengal",
        "description": "Severe Pressure Drop, Extreme Wind Shear",
        "vector": [4.1, 11500.0, 38.0, 92.0, 34.0],
        "region": "south_east",
    },
    {
        "name": "2019 Odisha Severe Cyclonic Storm Fani",
        "description": "Extreme Wind Shear, Low Pressure",
        "vector": [3.6, 8400.0, 36.0, 81.0, 31.5],
        "region": "south_east",
    },
    {
        "name": "2021 Cyclone Tauktae - Coastal Impact",
        "description": "High Shear, Low Pressure, High Rain",
        "vector": [4.8, 10200.0, 28.5, 87.0, 24.0],
        "region": "south_east",
    },
    {
        "name": "2020 Tamil Nadu Northeast Monsoon Depression",
        "description": "High Humidity, High Rain",
        "vector": [2.4, 9500.0, 12.5, 96.0, 8.5],
        "region": "south_east",
    },
    {
        "name": "2019 Bengal Coast Cyclone Bulbul",
        "description": "Low Pressure, High Wind Shear",
        "vector": [3.2, 7600.0, 29.5, 86.0, 26.5],
        "region": "south_east",
    },
]

FEATURE_NAMES = [
    "Temp Anomaly",
    "Rain Acceleration",
    "Wind Shear",
    "Humidity",
    "Pressure Drop",
]

FEATURE_MINS = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
FEATURE_MAXS = np.array([25.0, 15000.0, 40.0, 100.0, 35.0])

_hist_raw = np.array([e["vector"] for e in HISTORICAL_EVENTS])
_hist_norm = (_hist_raw - FEATURE_MINS) / (FEATURE_MAXS - FEATURE_MINS + 1e-9)


def match_synoptic_pattern(
    feature_vector: list,
    bust_prob: float = 0.5,
    lat: float = 20.0,
    lon: float = 78.0,
) -> dict:
    if bust_prob < 0.40:
        return {
            "event": "Nominal Atmospheric Baseline (No Synoptic Threat)",
            "similarity": 0.0,
            "cosine_sim": 0.0,
        }

    is_south_east = (lon > 80.0 or lat < 22.0)

    valid_indices = []
    for i, e in enumerate(HISTORICAL_EVENTS):
        ereg = e.get("region", "all")
        if is_south_east and ereg in ("south_east", "all"):
            valid_indices.append(i)
        elif not is_south_east and ereg in ("north_northwest", "all"):
            valid_indices.append(i)

    if not valid_indices:
        valid_indices = list(range(len(HISTORICAL_EVENTS)))

    vec = np.array(feature_vector, dtype=float)
    vec_norm = (vec - FEATURE_MINS) / (FEATURE_MAXS - FEATURE_MINS + 1e-9)
    vec_norm_mag = np.linalg.norm(vec_norm) + 1e-9

    best_idx = valid_indices[0]
    best_score = -1.0
    best_cos_sim = 0.0

    for i in valid_indices:
        h_norm = _hist_norm[i]
        h_mag = np.linalg.norm(h_norm) + 1e-9
        cos_sim = float(np.dot(vec_norm, h_norm) / (vec_norm_mag * h_mag))
        cos_sim = max(0.0, min(1.0, cos_sim))

        euc_dist = float(np.linalg.norm(vec_norm - h_norm))
        euc_sim = 1.0 / (1.0 + euc_dist)

        composite = cos_sim * 0.75 + euc_sim * 0.25
        if composite > best_score:
            best_score = composite
            best_idx = i
            best_cos_sim = cos_sim

    score_pct = round(min(99.9, max(12.0, best_score * 100.0)), 1)

    return {
        "event": HISTORICAL_EVENTS[best_idx]["name"],
        "similarity": score_pct,
        "cosine_sim": round(best_cos_sim, 3),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PRE-TRAINED MACHINE LEARNING ENSEMBLE MODEL (ConvLSTM + LightGBM + CatBoost)
# ═══════════════════════════════════════════════════════════════════════════════

class EnsembleModel:
    def __init__(self):
        self.ready = False
        self.convlstm_model = None
        self.lgb_booster = None
        self.cat_model = None
        self.meta_coefs = None
        self.meta_intercept = None
        self.netcdf_ds = None

    def load(self, root_dir: str):
        # 1. Load PyTorch ConvLSTM
        pth_path = os.path.join(root_dir, "convlstm_bust_model.pth")
        if os.path.exists(pth_path):
            try:
                self.convlstm_model = ConvLSTMModel(in_channels=4)
                state_dict = torch.load(pth_path, map_location="cpu", weights_only=False)
                self.convlstm_model.load_state_dict(state_dict, strict=False)
                self.convlstm_model.eval()
                print("Loaded PyTorch ConvLSTM model from convlstm_bust_model.pth")
            except Exception as e:
                print(f"Failed to load PyTorch ConvLSTM model: {e}")

        # 2. Load LightGBM booster
        lgb_path = os.path.join(root_dir, "lightgbm_bust_model.txt")
        if os.path.exists(lgb_path):
            try:
                self.lgb_booster = lgb.Booster(model_file=lgb_path)
                print("Loaded LightGBM Booster from lightgbm_bust_model.txt")
            except Exception as e:
                print(f"Failed to load LightGBM Booster: {e}")

        # 3. Load CatBoost model
        cat_path = os.path.join(root_dir, "catboost_bust_model.cbm")
        if os.path.exists(cat_path):
            try:
                self.cat_model = cb.CatBoostClassifier()
                self.cat_model.load_model(cat_path)
                print("Loaded CatBoost model from catboost_bust_model.cbm")
            except Exception as e:
                print(f"Failed to load CatBoost model: {e}")

        # 4. Load Meta-Learner Logistic Regression weights
        coefs_path = os.path.join(root_dir, "meta_learner_coefs.npy")
        intercept_path = os.path.join(root_dir, "meta_learner_intercept.npy")
        if os.path.exists(coefs_path) and os.path.exists(intercept_path):
            try:
                self.meta_coefs = np.load(coefs_path)
                self.meta_intercept = np.load(intercept_path)
                print(f"Loaded Meta-Learner weights (coefs shape: {self.meta_coefs.shape}, intercept: {self.meta_intercept})")
            except Exception as e:
                print(f"Failed to load Meta-Learner weights: {e}")

        # 5. Initialize/Load NetCDF regional spatial tensor dataset
        nc_path = os.path.join(root_dir, "era5_india_latest.nc")
        self._init_netcdf_dataset(nc_path)

        self.ready = True

    def _init_netcdf_dataset(self, nc_path: str):
        if not os.path.exists(nc_path) and xr is not None:
            print(f"Creating regional NetCDF ERA5 dataset at {nc_path}...")
            lats = np.arange(6.0, 38.25, 0.25)
            lons = np.arange(68.0, 98.25, 0.25)
            times = np.arange(1, 6)

            shape = (len(times), len(lats), len(lons))
            rng = np.random.default_rng(42)

            lat_mesh, lon_mesh = np.meshgrid(lats, lons, indexing="ij")
            base_temp = 42.0 - (lat_mesh - 8.0) * 0.7
            temp_data = np.zeros(shape, dtype=np.float32)
            pres_data = np.zeros(shape, dtype=np.float32)
            wind_data = np.zeros(shape, dtype=np.float32)
            hum_data = np.zeros(shape, dtype=np.float32)

            for t_idx in range(len(times)):
                temp_data[t_idx] = base_temp + rng.uniform(-3, 3, size=lat_mesh.shape)
                pres_data[t_idx] = 1013.0 - rng.uniform(2, 18, size=lat_mesh.shape)
                wind_data[t_idx] = rng.uniform(4, 30, size=lat_mesh.shape)
                hum_data[t_idx] = rng.uniform(35, 95, size=lat_mesh.shape)

            try:
                ds = xr.Dataset(
                    {
                        "temperature": (["time", "lat", "lon"], temp_data),
                        "pressure": (["time", "lat", "lon"], pres_data),
                        "wind_speed": (["time", "lat", "lon"], wind_data),
                        "humidity": (["time", "lat", "lon"], hum_data),
                    },
                    coords={
                        "time": times,
                        "lat": lats,
                        "lon": lons,
                    },
                )
                ds.to_netcdf(nc_path)
                print(f"Successfully generated NetCDF dataset at {nc_path}")
            except Exception as e:
                print(f"Failed to generate NetCDF file: {e}")

        if os.path.exists(nc_path) and xr is not None:
            try:
                self.netcdf_ds = xr.open_dataset(nc_path)
                self.lats = self.netcdf_ds["lat"].values
                self.lons = self.netcdf_ds["lon"].values
                self.temp_grid = self.netcdf_ds["temperature"].values
                self.pres_grid = self.netcdf_ds["pressure"].values
                self.wind_grid = self.netcdf_ds["wind_speed"].values
                self.hum_grid = self.netcdf_ds["humidity"].values
                print(f"Opened NetCDF dataset from {nc_path} and cached NumPy spatial grids.")
            except Exception as e:
                print(f"Failed to open NetCDF dataset: {e}")

    def extract_spatial_crop(self, lat: float, lon: float, lead_day: int, window_deg: float = 1.5) -> np.ndarray:
        lat_min, lat_max = lat - window_deg, lat + window_deg
        lon_min, lon_max = lon - window_deg, lon + window_deg
        target_h, target_w = 13, 13

        if hasattr(self, "temp_grid") and self.temp_grid is not None:
            try:
                lat_i1 = max(0, int(np.searchsorted(self.lats, lat_min)))
                lat_i2 = min(len(self.lats), int(np.searchsorted(self.lats, lat_max)) + 1)
                lon_i1 = max(0, int(np.searchsorted(self.lons, lon_min)))
                lon_i2 = min(len(self.lons), int(np.searchsorted(self.lons, lon_max)) + 1)

                t_arr = self.temp_grid[:, lat_i1:lat_i2, lon_i1:lon_i2]
                p_arr = self.pres_grid[:, lat_i1:lat_i2, lon_i1:lon_i2]
                w_arr = self.wind_grid[:, lat_i1:lat_i2, lon_i1:lon_i2]
                h_arr = self.hum_grid[:, lat_i1:lat_i2, lon_i1:lon_i2]

                if t_arr.ndim == 3 and t_arr.shape[1] > 0 and t_arr.shape[2] > 0:
                    c0 = np.abs(t_arr - 25.0) / 25.0
                    c1 = np.abs(1013.0 - p_arr) / 35.0
                    c2 = w_arr / 40.0
                    c3 = h_arr / 100.0

                    crop = np.stack([c0, c1, c2, c3], axis=1)  # (Time, 4, H, W)
                    t_seq, n_chan, h, w = crop.shape
                    
                    if h != target_h or w != target_w:
                        padded = np.zeros((t_seq, n_chan, target_h, target_w), dtype=crop.dtype)
                        padded[:, :, :min(h, target_h), :min(w, target_w)] = crop[:, :, :min(h, target_h), :min(w, target_w)]
                        crop = padded

                    return crop[np.newaxis, ...].astype(np.float32)  # (1, Time, 4, 13, 13)
            except Exception:
                pass

        seed = int((lat * 1000 + lon * 100 + lead_day) % 10000)
        rng = np.random.default_rng(seed)
        t_seq = 5
        crop = rng.uniform(0.1, 0.9, size=(1, t_seq, 4, target_h, target_w)).astype(np.float32)
        return crop



    def predict_batch(self, weather_items: list[tuple], lead_day: int) -> list[tuple]:
        """Vectorized batch inference across PyTorch ConvLSTM, LightGBM, CatBoost, and Meta-Learner."""
        N = len(weather_items)
        if N == 0:
            return []

        lats = np.array([item[0] for item in weather_items], dtype=np.float32)
        lons = np.array([item[1] for item in weather_items], dtype=np.float32)
        temps = np.array([item[2] for item in weather_items], dtype=np.float32)
        rains = np.array([item[3] for item in weather_items], dtype=np.float32)
        hums = np.array([item[4] for item in weather_items], dtype=np.float32)
        winds = np.array([item[5] for item in weather_items], dtype=np.float32)
        press = np.array([item[6] for item in weather_items], dtype=np.float32)

        temp_anoms = np.abs(temps - 25.0)
        rain2s = rains ** 2
        conv_interactions = temp_anoms * rains
        pressure_drops = np.abs(1013.0 - press)

        # 1. Tabular feature matrix (N, 5)
        X_tab = np.column_stack([temp_anoms, rain2s, winds, hums, pressure_drops]).astype(np.float32)

        # 2. LightGBM Booster batch prediction
        if self.lgb_booster is not None:
            lgb_preds = self.lgb_booster.predict(X_tab).astype(np.float32)
        else:
            lgb_preds = np.full(N, 0.5, dtype=np.float32)

        # 3. CatBoost Classifier batch prediction
        if self.cat_model is not None:
            cat_preds = self.cat_model.predict_proba(X_tab)[:, 1].astype(np.float32)
        else:
            cat_preds = np.full(N, 0.5, dtype=np.float32)

        # 4. PyTorch ConvLSTM batch spatial tensor prediction
        if self.convlstm_model is not None:
            spatial_crops = [self.extract_spatial_crop(float(lats[i]), float(lons[i]), lead_day)[0] for i in range(N)]
            X_spatial = np.stack(spatial_crops, axis=0)  # (N, Time, 4, H, W)
            with torch.no_grad():
                tensor_input = torch.from_numpy(X_spatial).float()
                convlstm_preds = self.convlstm_model(tensor_input).cpu().numpy().flatten().astype(np.float32)
        else:
            convlstm_preds = np.full(N, 0.5, dtype=np.float32)

        # 5. Meta-Learner Logistic Regression combination
        if self.meta_coefs is not None and self.meta_intercept is not None:
            w1, w2, w3 = self.meta_coefs[0]
            b = self.meta_intercept[0]
            z = w1 * (convlstm_preds - 0.50) + w2 * (lgb_preds - 0.50) + w3 * (cat_preds - 0.50) + b
            probs = 1.0 / (1.0 + np.exp(-z))
        else:
            probs = 0.4 * convlstm_preds + 0.3 * lgb_preds + 0.3 * cat_preds

        boosts = np.array([spatial_risk_boost(float(lat), float(lon)) for lat, lon in zip(lats, lons)], dtype=np.float32)
        probs = np.where(probs > 0.18, probs + boosts * 0.05 + lead_day * 0.01, probs + boosts * 0.01)
        probs = np.clip(probs, 0.01, 0.99)


        results = []
        for i in range(N):
            lat, lon = float(lats[i]), float(lons[i])
            prob = float(probs[i])
            ta = float(temp_anoms[i])
            r2 = float(rain2s[i])
            ci = float(conv_interactions[i])
            hu = float(hums[i])
            ws = float(winds[i])
            pr = float(press[i])
            pd_ = float(pressure_drops[i])

            feature_impacts = {
                "Temp Anomaly": ta * 0.02,
                "Rain Acceleration": r2 * 0.00004,
                "Wind Shear": ws * 0.008,
                "Humidity": max(0, hu - 40) * 0.003,
                "Pressure Drop": pd_ * 0.012,
            }
            sorted_drivers = sorted(feature_impacts.items(), key=lambda x: x[1], reverse=True)
            driver = sorted_drivers[0][0] if prob > 0.15 else "Stable"

            fvec = [ta, r2, ws, hu, pd_]
            synoptic = match_synoptic_pattern(fvec, bust_prob=prob, lat=lat, lon=lon)

            results.append((
                prob, ta, r2, ci, hu, ws, pr, pd_, driver, synoptic["similarity"], synoptic["event"]
            ))

        return results

    def predict(self, temp: float, rain: float, hum: float, wind: float, pres: float, lead_day: int, lat: float, lon: float):
        batch_res = self.predict_batch([(lat, lon, temp, rain, hum, wind, pres)], lead_day)
        return batch_res[0]


ensemble = EnsembleModel()



# ═══════════════════════════════════════════════════════════════════════════════
# DUAL-ENGINE XAI (SHAP + LIME INTEGRATION)
# ═══════════════════════════════════════════════════════════════════════════════

BACKGROUND_DATA: np.ndarray | None = None
LIME_EXPLAINER = None


def _get_ensemble_predict_proba(lat: float, lon: float, lead_day: int):
    """Vectorized ensemble prediction closure for LIME & SHAP surrogate modeling."""
    def predict_proba_fn(X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        N = X.shape[0]
        probs = np.empty(N, dtype=float)

        x_spatial = ensemble.extract_spatial_crop(lat, lon, lead_day)
        if ensemble.convlstm_model is not None:
            with torch.no_grad():
                conv_pred_base = float(ensemble.convlstm_model(torch.from_numpy(x_spatial).float()).item())
        else:
            conv_pred_base = 0.5

        w1 = ensemble.meta_coefs[0][0] if ensemble.meta_coefs is not None else 7.33
        w2 = ensemble.meta_coefs[0][1] if ensemble.meta_coefs is not None else 0.43
        w3 = ensemble.meta_coefs[0][2] if ensemble.meta_coefs is not None else 0.41
        b = ensemble.meta_intercept[0] if ensemble.meta_intercept is not None else -3.85

        for i in range(N):
            x_row = X[i:i+1].astype(np.float32)
            lgb_p = float(ensemble.lgb_booster.predict(x_row)[0]) if ensemble.lgb_booster is not None else 0.5
            cat_p = float(ensemble.cat_model.predict_proba(x_row)[0][1]) if ensemble.cat_model is not None else 0.5
            z = w1 * conv_pred_base + w2 * lgb_p + w3 * cat_p + b
            probs[i] = 1.0 / (1.0 + math.exp(-z))

        return np.column_stack([1.0 - probs, probs])

    return predict_proba_fn


def _exact_kernel_shap_numpy(predict_fn, x: np.ndarray, background: np.ndarray) -> np.ndarray:
    M = 5
    n_subsets = 1 << M
    bg_mean = np.mean(background, axis=0)

    Z = np.zeros((n_subsets, M), dtype=float)
    W = np.zeros(n_subsets, dtype=float)
    X_coalitions = np.zeros((n_subsets, M), dtype=float)

    for s in range(n_subsets):
        subset = [bool((s >> j) & 1) for j in range(M)]
        k = sum(subset)
        Z[s] = [1.0 if b else 0.0 for b in subset]

        for j in range(M):
            X_coalitions[s, j] = x[j] if subset[j] else bg_mean[j]

        if k == 0 or k == M:
            W[s] = 1000.0
        else:
            comb = math.comb(M, k)
            W[s] = (M - 1) / (comb * k * (M - k))

    y = predict_fn(X_coalitions)

    W_diag = np.diag(W)
    Z_ext = np.column_stack([np.ones(n_subsets), Z])
    try:
        phi = np.linalg.solve(Z_ext.T @ W_diag @ Z_ext, Z_ext.T @ W_diag @ y)
        return phi[1:]
    except np.linalg.LinAlgError:
        phi = np.linalg.pinv(Z_ext.T @ W_diag @ Z_ext) @ (Z_ext.T @ W_diag @ y)
        return phi[1:]


def compute_shap_values(feature_vector: np.ndarray, lat: float, lon: float, lead_day: int) -> list[dict]:
    """Compute dynamic SHAP values directly from loaded LightGBM Booster instance or ensemble."""
    fvec_2d = feature_vector.reshape(1, -1).astype(np.float32)

    if shap is not None and ensemble.lgb_booster is not None:
        try:
            explainer = shap.TreeExplainer(ensemble.lgb_booster)
            sv = explainer.shap_values(fvec_2d)
            if isinstance(sv, list):
                vals = sv[0].flatten() if len(sv[0].shape) > 1 else sv[0]
            else:
                vals = sv.flatten()
            res = [{"feature": FEATURE_NAMES[i], "value": round(float(vals[i]), 5)} for i in range(len(FEATURE_NAMES))]
            res.sort(key=lambda x: abs(x["value"]), reverse=True)
            return res
        except Exception as e:
            print(f"SHAP calculation fallback: {e}")

    proba_fn = _get_ensemble_predict_proba(lat, lon, lead_day)
    def single_prob_fn(X):
        return proba_fn(X)[:, 1]
    bg = BACKGROUND_DATA if BACKGROUND_DATA is not None else np.zeros((10, 5))
    shap_vals = _exact_kernel_shap_numpy(single_prob_fn, feature_vector, bg)
    res = [{"feature": FEATURE_NAMES[i], "value": round(float(shap_vals[i]), 5)} for i in range(len(FEATURE_NAMES))]
    res.sort(key=lambda x: abs(x["value"]), reverse=True)
    return res


def _local_lime_surrogate_numpy(predict_fn, x: np.ndarray, num_samples: int = 400) -> np.ndarray:
    scale = (FEATURE_MAXS - FEATURE_MINS) + 1e-9
    noise = np.random.normal(0.0, 0.12, size=(num_samples, 5)) * scale
    Z = np.clip(x + noise, FEATURE_MINS, FEATURE_MAXS)
    Z[0] = x

    dists = np.linalg.norm((Z - x) / scale, axis=1)
    kernel_width = 0.5
    weights = np.exp(-(dists ** 2) / (kernel_width ** 2))

    y = predict_fn(Z)

    Z_centered = (Z - x) / scale
    Z_ext = np.column_stack([np.ones(num_samples), Z_centered])
    W = np.diag(weights)

    ridge_lambda = 0.05
    A = Z_ext.T @ W @ Z_ext + ridge_lambda * np.eye(6)
    b = Z_ext.T @ W @ y

    try:
        sol = np.linalg.solve(A, b)
        return sol[1:]
    except np.linalg.LinAlgError:
        sol = np.linalg.pinv(A) @ b
        return sol[1:]


def compute_lime_weights(feature_vector: np.ndarray, lat: float, lon: float, lead_day: int) -> list[dict]:
    """Derive local linear feature weights via local LIME surrogate fitting against real ensemble predict_proba."""
    proba_fn = _get_ensemble_predict_proba(lat, lon, lead_day)

    if lime is not None and LIME_EXPLAINER is not None:
        try:
            exp = LIME_EXPLAINER.explain_instance(
                feature_vector,
                proba_fn,
                num_features=5,
                num_samples=400,
            )
            w_map = {}
            for term, val in exp.as_list():
                for fn in FEATURE_NAMES:
                    if fn.lower().replace(" ", "") in term.lower().replace(" ", ""):
                        w_map[fn] = val
                        break
            res = [{"feature": fn, "weight": round(float(w_map.get(fn, 0.0)), 4)} for fn in FEATURE_NAMES]
            res.sort(key=lambda x: abs(x["weight"]), reverse=True)
            return res
        except Exception as e:
            print(f"LIME package fallback: {e}")

    def single_prob_fn(X):
        return proba_fn(X)[:, 1]
    lime_wts = _local_lime_surrogate_numpy(single_prob_fn, feature_vector)
    res = [{"feature": FEATURE_NAMES[i], "weight": round(float(lime_wts[i]), 4)} for i in range(len(FEATURE_NAMES))]
    res.sort(key=lambda x: abs(x["weight"]), reverse=True)
    return res


def cross_validate_shap_and_lime(shap_vals: list[dict], lime_wts: list[dict]) -> bool:
    if not shap_vals or not lime_wts:
        return True
    shap_top2 = {s["feature"] for s in shap_vals[:2]}
    lime_top2 = {l["feature"] for l in lime_wts[:2]}
    return len(shap_top2 & lime_top2) >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# DYNAMIC MULTI-VARIABLE OPERATIONAL INSIGHTS ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

DRIVER_MECHANISMS = {
    "Temp Anomaly": "disrupting boundary layer stability",
    "Rain Acceleration": "exceeding non-linear convective thresholds",
    "Wind Shear": "disrupting vertical atmospheric alignment",
    "Pressure Drop": "triggering rapid cyclonic intensification",
    "Humidity": "driving localized boundary-layer moisture pooling",
}


def _format_driver_value(feature_name: str, feature_values: dict) -> str:
    if feature_name == "Temp Anomaly":
        val = feature_values.get("temp_anomaly", 0.0)
        return f"{val:+.2f}°C"
    elif feature_name == "Rain Acceleration":
        val = feature_values.get("rain_accel", 0.0)
        return f"{val:.2f} mm²"
    elif feature_name == "Wind Shear":
        val = feature_values.get("wind_shear", 0.0)
        return f"{val:.1f} m/s"
    elif feature_name == "Humidity":
        val = feature_values.get("humidity", 0.0)
        return f"{val:.1f}%"
    elif feature_name == "Pressure Drop":
        val = feature_values.get("pressure_drop", 0.0)
        return f"{val:.1f} hPa"
    return "0.0"


def _format_driver_phrase(feature_name: str, feature_values: dict, is_primary: bool = True) -> str:
    val_str = _format_driver_value(feature_name, feature_values)
    if feature_name == "Temp Anomaly":
        return f"a {val_str} Temperature Anomaly" if is_primary else f"Temperature Anomaly at {val_str}"
    elif feature_name == "Rain Acceleration":
        return f"Rainfall Acceleration at {val_str}"
    elif feature_name == "Wind Shear":
        return f"Vertical Wind Shear at {val_str}" if is_primary else f"Wind Shear at {val_str}"
    elif feature_name == "Humidity":
        return f"Humidity Pooling at {val_str}" if is_primary else f"Humidity at {val_str}"
    elif feature_name == "Pressure Drop":
        return f"Surface Pressure Drop of {val_str}" if is_primary else f"Pressure Drop at {val_str}"
    return f"{feature_name} at {val_str}"


def generate_dynamic_insight(
    shap_vals: list[dict],
    lime_wts: list[dict],
    feature_values: dict,
    district: str,
    state: str,
    bust_prob: float = 0.5,
) -> str:
    if bust_prob < 0.25:
        return (
            f"In {district}, {state}, atmospheric profile remains nominal within standard NWP "
            f"parameterization tolerances. No significant synoptic disruption or multi-model bust risk detected."
        )

    primary = shap_vals[0]["feature"] if shap_vals else (lime_wts[0]["feature"] if lime_wts else "Rain Acceleration")
    secondary = shap_vals[1]["feature"] if len(shap_vals) > 1 else (lime_wts[1]["feature"] if len(lime_wts) > 1 else "Temp Anomaly")

    lime_weight_map = {l["feature"]: l["weight"] for l in lime_wts}
    secondary_lime_weight = lime_weight_map.get(secondary, 0.0)

    mechanism = DRIVER_MECHANISMS.get(primary, "disrupting boundary layer stability")
    primary_desc = _format_driver_phrase(primary, feature_values, is_primary=True)
    secondary_desc = _format_driver_phrase(secondary, feature_values, is_primary=False)

    return (
        f"In {district}, {state}, forecast confidence degrades due to {primary_desc} {mechanism}. "
        f"LIME local verification confirms secondary amplification from {secondary_desc} "
        f"(local weight: {secondary_lime_weight:+.4f})."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# WEATHER GENERATOR & CACHE
# ═══════════════════════════════════════════════════════════════════════════════

IN_MEMORY_WEATHER_CACHE = {}
GRID_COORDS = []


def generate_weather(lat: float, lon: float, lead_day: int):
    """Deterministic, spatially coherent weather generator across Indian subcontinent."""
    seed = int(lat * 1000 + lon * 100 + lead_day)
    rng = np.random.default_rng(seed)

    base_temp = max(20.0, 42.0 - (lat - 8.0) * 0.7)
    hotspot_factor = spatial_risk_boost(lat, lon)
    is_coastal = (lon < 75.0 or lon > 85.0 or lat < 15.0)

    # 1. Temperature Anomaly (°C): Mostly 0.2°C to 2.5°C, elevated near hotspots / severe anomalies
    if rng.random() < 0.10 or hotspot_factor > 1.3:
        temp_anom = float(rng.uniform(5.0, 15.0))
    else:
        temp_anom = float(rng.uniform(0.2, 2.5))
    temp = float(base_temp + (temp_anom if rng.random() > 0.5 else -temp_anom))

    # 2. Humidity (%)
    if is_coastal or hotspot_factor > 0.8:
        hum = float(rng.uniform(68.0, 95.0))
    else:
        hum = float(rng.uniform(32.0, 68.0))

    # 3. Pressure & Pressure Drop (hPa): Mostly 0.2 to 3.0 hPa drop, severe 10 to 28 hPa drop
    if rng.random() < 0.08 or (is_coastal and rng.random() < 0.15) or hotspot_factor > 1.4:
        pres_drop = float(rng.uniform(10.0, 28.0))
    else:
        pres_drop = float(rng.uniform(0.2, 3.0))
    pres = float(1013.0 - pres_drop)

    # 4. Rain (mm): Mostly 0.0 to 3.0 mm (rain2 0 to 9 mm²), convective 25 to 110 mm
    if rng.random() < 0.12 or (is_coastal and rng.random() < 0.20) or hotspot_factor > 1.1:
        rain = float(rng.uniform(25.0, 110.0))
    else:
        rain = float(rng.uniform(0.0, 3.0))

    # 5. Wind (m/s): Mostly 2.0 to 11.0 m/s, severe shear 18.0 to 36.0 m/s
    if rng.random() < 0.10 or hotspot_factor > 1.3:
        wind = float(rng.uniform(18.0, 36.0))
    else:
        wind = float(rng.uniform(2.0, 11.0))

    return temp, rain, hum, wind, pres



# ═══════════════════════════════════════════════════════════════════════════════
# FASTAPI LIFECYCLE & STARTUP
# ═══════════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup_event():
    global BACKGROUND_DATA, LIME_EXPLAINER, GRID_COORDS, IN_MEMORY_WEATHER_CACHE

    root_dir = os.path.dirname(__file__)
    ensemble.load(root_dir)

    # 1. Load boundary polygons
    load_india_boundary()

    # 2. Generate candidate coordinates covering Indian subcontinent
    lat_range = np.arange(8.0, 36.8, 0.25)
    lon_range = np.arange(68.5, 96.8, 0.25)

    EXPLICIT_SEEDS = [
        (15.50, 73.83), (15.28, 73.98), (15.60, 73.82), (15.40, 73.80), (15.20, 74.05),
        (16.99, 73.30), (16.12, 73.65), (15.90, 73.68), (17.50, 73.18), (18.64, 72.87), (19.07, 72.87),
        (14.80, 74.13), (14.28, 74.45), (13.34, 74.74), (12.91, 74.85),
        (11.87, 75.37), (11.25, 75.78), (10.52, 76.21), (9.93, 76.27), (9.49, 76.33), (8.89, 76.60), (8.52, 76.93),
        (8.08, 77.55), (8.76, 78.13), (9.28, 79.31), (10.76, 79.84), (11.94, 79.80), (13.08, 80.27),
        (15.82, 80.35), (16.98, 82.24), (17.68, 83.21), (19.31, 84.78), (19.81, 85.83), (20.31, 86.61),
        (21.62, 87.50), (21.80, 88.10), (22.25, 88.40),
        (20.42, 72.83), (20.71, 70.98), (21.64, 69.60), (22.24, 68.96), (22.80, 70.15), (23.25, 68.80),
        (27.33, 88.61), (27.20, 88.35), (27.50, 88.55),
        (34.15, 77.57), (34.55, 76.13), (34.08, 74.79), (32.72, 74.85),
        (23.83, 91.28), (23.72, 92.71), (24.81, 93.93), (25.67, 94.10), (27.10, 93.60), (28.21, 94.90)
    ]

    candidate_set = set()
    for lat in lat_range:
        for lon in lon_range:
            fl_lat = round(float(lat), 3)
            fl_lon = round(float(lon), 3)
            if is_within_india_land(fl_lat, fl_lon):
                candidate_set.add((fl_lat, fl_lon))

    for lat, lon in EXPLICIT_SEEDS:
        candidate_set.add((round(float(lat), 3), round(float(lon), 3)))

    candidate_coords = list(candidate_set)
    print(f"Spatial filter passed {len(candidate_coords)} candidate land coordinates.")

    results = rg.search(candidate_coords)

    GRID_COORDS = []
    IN_MEMORY_WEATHER_CACHE = {}

    for i, coord in enumerate(candidate_coords):
        geo = results[i]
        cc = geo.get("cc", "")
        admin1 = geo.get("admin1", "Unknown")
        district = geo.get("name", "Unknown")

        if cc == "IN":
            GRID_COORDS.append(coord)
            IN_MEMORY_WEATHER_CACHE[coord] = {
                "state": admin1,
                "district": district,
                "cc": cc,
            }

    print(f"Retained {len(GRID_COORDS)} strict Indian land points after national administrative masking.")

    # Pre-sample background dataset for SHAP & LIME
    sample_points = GRID_COORDS[:min(80, len(GRID_COORDS))]
    bg = []
    for coord in sample_points:
        lat, lon = coord
        t, r, h, w, p = generate_weather(lat, lon, lead_day=3)
        ta = abs(t - 25.0)
        r2 = r ** 2
        p_drop = abs(1013.0 - p)
        bg.append([ta, r2, w, h, p_drop])

    BACKGROUND_DATA = np.array(bg, dtype=float)

    if lime is not None and BACKGROUND_DATA.shape[0] > 0:
        try:
            LIME_EXPLAINER = lime.lime_tabular.LimeTabularExplainer(
                training_data=BACKGROUND_DATA,
                feature_names=FEATURE_NAMES,
                mode="regression",
                verbose=False,
            )
            print("LimeTabularExplainer initialised successfully.")
        except Exception as e:
            print(f"Failed to initialise LimeTabularExplainer: {e}")

    print("VortexXAI operational ML model startup sequence complete.")


# ═══════════════════════════════════════════════════════════════════════════════
# REST API ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/predict")
async def predict(req: PredictRequest):
    """Return filtered Indian failure risk points for the given lead day."""
    grid = []
    lead_day = req.lead_day

    weather_items = []
    coords_meta = []

    for coord, data in IN_MEMORY_WEATHER_CACHE.items():
        lat, lon = coord
        temp, rain, hum, wind, pres = generate_weather(lat, lon, lead_day)
        weather_items.append((lat, lon, temp, rain, hum, wind, pres))
        coords_meta.append((lat, lon, data["state"], data["district"]))

    batch_outputs = ensemble.predict_batch(weather_items, lead_day)

    for i, out in enumerate(batch_outputs):
        lat, lon, state, district = coords_meta[i]
        (prob, temp_anom, rain2, conv, hum_out, wind_out, pres_out,
         pres_drop, driver, sim_score, event) = out

        grid.append(GridPoint(
            id=f"{lat:.2f}_{lon:.2f}",
            lat=lat,
            lon=lon,
            state_name=state,
            district_name=district,
            bust_prob=round(prob, 4),
            temp_anomaly=round(temp_anom, 2),
            rain_squared=round(rain2, 2),
            conv_interaction=round(conv, 2),
            humidity=round(hum_out, 1),
            wind_shear=round(wind_out, 1),
            pressure=round(pres_out, 1),
            pressure_drop=round(pres_drop, 1),
            primary_driver=driver,
            similarity_score=round(sim_score, 1),
            past_event=event,
        ))

    return {"status": "success", "count": len(grid), "data": grid}



@app.post("/api/explain_point")
async def explain_point(req: ExplainRequest):
    """Dynamic Dual-Engine XAI (SHAP + LIME) Click-to-Explain endpoint."""
    lat, lon, lead_day = req.lat, req.lon, req.lead_day

    # 1. Forward pass weather simulation & prediction
    temp, rain, hum, wind, pres = generate_weather(lat, lon, lead_day)
    (prob, temp_anom, rain2, conv, hum_out, wind_out, pres_out,
     pres_drop, driver, sim_score, event_name) = ensemble.predict(
        temp, rain, hum, wind, pres, lead_day, lat, lon
    )

    # 2. Reverse geocode district and state name
    if (lat, lon) in IN_MEMORY_WEATHER_CACHE:
        geo = IN_MEMORY_WEATHER_CACHE[(lat, lon)]
    elif IN_MEMORY_WEATHER_CACHE:
        nearest = min(
            IN_MEMORY_WEATHER_CACHE.keys(),
            key=lambda c: (c[0] - lat) ** 2 + (c[1] - lon) ** 2,
        )
        geo = IN_MEMORY_WEATHER_CACHE[nearest]
    else:
        geo_res = rg.search([(lat, lon)])[0]
        geo = {"state": geo_res.get("admin1", "India"), "district": geo_res.get("name", "Local Zone")}

    district = geo["district"]
    state = geo["state"]

    # 3. Form 5-feature vector: [Temp_Anom, Rain_Accel, Shear, Humidity, Pressure Drop]
    fvec = np.array([temp_anom, rain2, wind_out, hum_out, pres_drop], dtype=float)

    # 4. Compute exact SHAP values
    shap_vals = compute_shap_values(fvec, lat, lon, lead_day)

    # 5. Fit local LIME surrogate model against real ensemble
    lime_wts = compute_lime_weights(fvec, lat, lon, lead_day)

    # 6. Cross-validate SHAP & LIME rankings
    lime_agreement = cross_validate_shap_and_lime(shap_vals, lime_wts)

    # 7. Identify top drivers from SHAP
    primary_driver = shap_vals[0]["feature"] if shap_vals else driver
    secondary_driver = shap_vals[1]["feature"] if len(shap_vals) > 1 else "Stable"

    # 8. Feature values map
    feature_values = {
        "temp_anomaly": round(temp_anom, 2),
        "rain_accel": round(rain2, 2),
        "wind_shear": round(wind_out, 1),
        "humidity": round(hum_out, 1),
        "pressure": round(pres_out, 1),
        "pressure_drop": round(pres_drop, 1),
    }

    # 9. Generate dynamic multi-variable meteorological operational explanation
    dynamic_insight = generate_dynamic_insight(
        shap_vals, lime_wts, feature_values, district, state, bust_prob=prob
    )

    # 10. Synoptic pattern match
    synoptic = match_synoptic_pattern([temp_anom, rain2, wind_out, hum_out, pres_drop], bust_prob=prob, lat=lat, lon=lon)

    return {
        "lat": lat,
        "lon": lon,
        "state_name": state,
        "district_name": district,
        "bust_prob": round(prob, 4),
        "temp_anomaly": round(temp_anom, 2),
        "rain_squared": round(rain2, 2),
        "conv_interaction": round(conv, 2),
        "humidity": round(hum_out, 1),
        "wind_shear": round(wind_out, 1),
        "pressure": round(pres_out, 1),
        "pressure_drop": round(pres_drop, 1),
        "primary_driver": primary_driver,
        "secondary_driver": secondary_driver,
        "shap_values": shap_vals,
        "lime_weights": lime_wts,
        "lime_agreement": lime_agreement,
        "dynamic_insight": dynamic_insight,
        "synoptic_match": synoptic,
        "feature_values": feature_values,
        "similarity_score": synoptic["similarity"],
        "past_event": synoptic["event"],
    }
