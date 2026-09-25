import { useEffect, useState } from "react";
import DeckGL from "@deck.gl/react";
import { ColumnLayer, ScatterplotLayer } from "@deck.gl/layers";
import Map from "react-map-gl/maplibre";
// @ts-ignore
import "maplibre-gl/dist/maplibre-gl.css";
import { Camera } from "lucide-react";

export interface GridPoint {
  id: string;
  lat: number;
  lon: number;
  latitude?: number;
  longitude?: number;
  district_name: string;
  state_name: string;
  bust_prob: number;
  primary_driver?: string;
  shap_analysis?: Array<{ feature_name: string; importance: number }>;
  shap_values?: Array<{ feature: string; value: number }>;
  [key: string]: any;
}

const MAP_STYLE =
  "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";

const INITIAL_VIEW_STATE = {
  longitude: 82.0,
  latitude: 22.0,
  zoom: 4.2,
  pitch: 55,
  bearing: 0,
};

const PRESETS = [
  { name: "🇮🇳  Full India", state: { ...INITIAL_VIEW_STATE } },
  {
    name: "🏔️  Himalayas",
    state: {
      longitude: 78.0,
      latitude: 30.0,
      zoom: 5.5,
      pitch: 60,
      bearing: -15,
    },
  },
  {
    name: "🌧️  Western Ghats",
    state: { longitude: 74.5, latitude: 14.0, zoom: 6, pitch: 50, bearing: 10 },
  },
  {
    name: "🌊  Bay of Bengal",
    state: {
      longitude: 86.0,
      latitude: 18.0,
      zoom: 5.5,
      pitch: 45,
      bearing: -20,
    },
  },
];

interface MapProps {
  data: GridPoint[];
  selectedGridId: string | null;
  onSelectGrid: (grid: GridPoint) => void;
  focusPoint?: { lat: number; lon: number } | null;
}

const getLon = (d: GridPoint): number => Number(d.longitude ?? d.lon ?? 0.0);
const getLat = (d: GridPoint): number => Number(d.latitude ?? d.lat ?? 0.0);

export default function MapComponent({
  data,
  selectedGridId,
  onSelectGrid,
  focusPoint,
}: MapProps) {
  const [viewState, setViewState] = useState(INITIAL_VIEW_STATE);
  const [hoverInfo, setHoverInfo] = useState<any>(null);

  // Find currently selected data point
  const selectedObject = data.find((d) => d.id === selectedGridId);

  useEffect(() => {
    if (!selectedObject) return;

    setViewState((currentView) => ({
      ...currentView,
      longitude: getLon(selectedObject),
      latitude: getLat(selectedObject),
      zoom: Math.min(Math.max(currentView.zoom + 1.25, 5.5), 7.5),
      transitionDuration: 700,
    }));
  }, [selectedGridId]);

  useEffect(() => {
    if (!focusPoint) return;
    setViewState((currentView) => ({
      ...currentView,
      longitude: focusPoint.lon,
      latitude: focusPoint.lat,
      zoom: Math.max(currentView.zoom, 6.2),
      transitionDuration: 900,
    }));
  }, [focusPoint]);

  const getColor = (
    prob: number,
    isSelected: boolean,
  ): [number, number, number, number] => {
    if (isSelected) return [255, 255, 255, 255];
    if (prob > 0.5) return [239, 68, 68, 230]; // Crimson Red
    if (prob > 0.3) return [245, 158, 11, 210]; // Amber
    return [16, 185, 129, 150]; // Emerald Green
  };

  const getElevation = (prob: number): number => {
    if (prob <= 0.3) return 2500;
    if (prob <= 0.5) return Math.pow(prob, 1.5) * 120000;
    return Math.pow(prob, 2) * 350000;
  };

  const layers: any[] = [
    // Base 3D Meteorological Cylinder / Column Layer (Optimized base radius & zoom scaling)
    new ColumnLayer({
      id: "bust-probability",
      data,
      diskResolution: 24,
      radius: 7500,
      radiusMinPixels: 3,
      radiusMaxPixels: 15,
      extruded: true,
      pickable: true,
      autoHighlight: false,
      elevationScale: 1,
      getPosition: (d: GridPoint) => [getLon(d), getLat(d)] as [number, number],
      getFillColor: (d: GridPoint) =>
        getColor(d.bust_prob, d.id === selectedGridId),
      getElevation: (d: GridPoint) => getElevation(d.bust_prob),
      onHover: (info: any) => setHoverInfo(info?.object ? info : null),
      onClick: (info: any) => {
        if (info.object) {
          onSelectGrid(info.object);
        }
      },
      updateTriggers: {
        getFillColor: [selectedGridId],
        getElevation: [data],
        getPosition: [data],
      },
    }),
  ];

  // Visual Anchors for Selected Grid Point
  if (selectedObject) {
    // 1. Crisp Cyan Glow Ring at column base
    layers.push(
      new ScatterplotLayer({
        id: "selection-ring",
        data: [selectedObject],
        pickable: false,
        stroked: true,
        filled: true,
        lineWidthMinPixels: 2.5,
        radiusMinPixels: 8,
        radiusMaxPixels: 24,
        getPosition: (d: GridPoint) =>
          [getLon(d), getLat(d)] as [number, number],
        getRadius: 12000,
        getFillColor: [6, 182, 212, 60] as [number, number, number, number],
        getLineColor: [6, 182, 212, 255] as [number, number, number, number],
        getLineWidth: 1800,
        updateTriggers: {
          getPosition: [selectedObject],
        },
      }),
    );

    // 2. White Cylinder Cap on top of selected 3D column
    layers.push(
      new ColumnLayer({
        id: "selection-cap",
        data: [selectedObject],
        pickable: false,
        diskResolution: 24,
        radius: 8500,
        radiusMinPixels: 5,
        radiusMaxPixels: 18,
        extruded: true,
        stroked: true,
        lineWidthMinPixels: 2,
        elevationScale: 1,
        getPosition: (d: GridPoint) =>
          [getLon(d), getLat(d)] as [number, number],
        getElevation: (d: GridPoint) => getElevation(d.bust_prob) + 2000,
        getFillColor: [255, 255, 255, 255] as [number, number, number, number],
        getLineColor: [6, 182, 212, 255] as [number, number, number, number],
        updateTriggers: {
          getPosition: [selectedObject],
          getElevation: [selectedObject],
        },
      }),
    );
  }

  return (
    <div className="w-full h-full relative overflow-hidden select-none">
      <DeckGL
        layers={layers}
        viewState={viewState}
        onViewStateChange={({ viewState }: any) => {
          const { transitionDuration, ...rest } = viewState;
          setViewState(rest);
        }}
        controller={true}
      >
        <Map mapStyle={MAP_STYLE} />
      </DeckGL>

      {/* Tooltip Popup Card - Pegged directly above the point anchor on screen */}
      {hoverInfo &&
        hoverInfo.object &&
        (() => {
          const obj: GridPoint = hoverInfo.object;
          const primaryDriver =
            obj.shap_analysis?.[0]?.feature_name ||
            obj.shap_values?.[0]?.feature ||
            obj.primary_driver ||
            "Stable";

          const lonVal = getLon(obj);
          const latVal = getLat(obj);
          const riskPct = obj.bust_prob * 100;
          const confidencePct = 100 - riskPct;
          const confidenceColor =
            confidencePct >= 70
              ? "#34D399"
              : confidencePct >= 40
                ? "#FBBF24"
                : "#F87171";

          return (
            <div
              className="absolute z-50 bg-slate-900/95 backdrop-blur-xl border border-slate-700/80 shadow-2xl p-4 rounded-xl pointer-events-none transition-transform duration-75"
              style={{
                left: hoverInfo.x,
                top: Math.max(16, hoverInfo.y - 12),
                transform: "translate(-50%, -100%)",
                minWidth: 220,
              }}
            >
              <div className="font-bold text-white text-[14.5px] mb-1.5 border-b border-slate-700/80 pb-1.5 flex items-center justify-between">
                <span>
                  {obj.district_name}, {obj.state_name}
                </span>
              </div>
              <div className="text-slate-300 text-xs font-mono">
                {latVal.toFixed(2)}°N, {lonVal.toFixed(2)}°E
              </div>
              <div className="mt-2.5 flex items-end gap-2">
                <div className="flex-1 min-w-0">
                  <div
                    className="text-xl font-extrabold font-mono"
                    style={{
                      color:
                        obj.bust_prob > 0.5
                          ? "#EF4444"
                          : obj.bust_prob > 0.3
                            ? "#F59E0B"
                            : "#10B981",
                    }}
                  >
                    {riskPct.toFixed(1)}%
                  </div>
                  <div className="text-[11px] font-semibold text-slate-300">
                    Risk
                  </div>
                </div>
                <div className="flex-1 min-w-0">
                  <div
                    className="text-xl font-extrabold font-mono"
                    style={{ color: confidenceColor }}
                  >
                    {confidencePct.toFixed(1)}%
                  </div>
                  <div className="text-[11px] font-semibold text-slate-300 whitespace-nowrap">
                    NWP Confidence
                  </div>
                </div>
              </div>
              <div className="text-slate-300 text-xs mt-2 font-medium flex items-center space-x-1.5 bg-slate-800/80 px-2.5 py-1 rounded-md border border-slate-700/50">
                <span className="text-slate-400">Driver:</span>
                <span className="text-cyan-400 font-bold">{primaryDriver}</span>
              </div>
              <div className="text-cyan-400 text-xs mt-2 font-semibold flex items-center justify-end">
                Click to analyze →
              </div>
            </div>
          );
        })()}

      {/* Camera Presets (User view presets) */}
      <div className="absolute bottom-6 left-6 z-10 flex flex-col space-y-2">
        <div className="text-xs font-bold text-slate-400 uppercase flex items-center mb-1 tracking-wider">
          <Camera className="w-4 h-4 mr-1.5" /> View Presets
        </div>
        {PRESETS.map((p, i) => (
          <button
            key={i}
            onClick={() =>
              setViewState({ ...p.state, transitionDuration: 600 } as any)
            }
            className="text-left px-3.5 py-1.5 text-xs font-semibold bg-slate-800/80 backdrop-blur border border-white/10 hover:bg-cyan-500/20 hover:border-cyan-500/50 transition-all text-white rounded-lg"
          >
            {p.name}
          </button>
        ))}
      </div>
    </div>
  );
}
