import { useState, useEffect, useCallback } from "react";
import {
  Group as PanelGroup,
  Panel,
  Separator as PanelResizeHandle,
  usePanelRef,
} from "react-resizable-panels";
import {
  AlertTriangle,
  Server,
  Zap,
  Maximize2,
  Sliders,
  ChevronUp,
  ChevronDown,
  Search,
} from "lucide-react";
import MapComponent from "./components/MapComponent";
import XAIDashboard from "./components/XAIDashboard";

const CITY_LOCATIONS = [
  ["Delhi", 28.6139, 77.209],
  ["Mumbai", 19.076, 72.8777],
  ["Pune", 18.5204, 73.8567],
  ["Kochi", 9.9312, 76.2673],
  ["Jamshedpur", 22.8046, 86.2029],
  ["Bengaluru", 12.9716, 77.5946],
  ["Hyderabad", 17.385, 78.4867],
  ["Chennai", 13.0827, 80.2707],
  ["Kolkata", 22.5726, 88.3639],
  ["Ahmedabad", 23.0225, 72.5714],
  ["Jaipur", 26.9124, 75.7873],
  ["Lucknow", 26.8467, 80.9462],
  ["Patna", 25.5941, 85.1376],
  ["Bhopal", 23.2599, 77.4126],
  ["Bhubaneswar", 20.2961, 85.8245],
  ["Guwahati", 26.1445, 91.7362],
  ["Srinagar", 34.0837, 74.7973],
  ["Dehradun", 30.3165, 78.0322],
  ["Shimla", 31.1048, 77.1734],
  ["Goa", 15.2993, 74.124],
  ["Juinagar, Navi Mumbai", 19.0475, 73.0169],
  ["Navi Mumbai", 19.033, 73.0297],
  ["Thane", 19.2183, 72.9781],
  ["Nagpur", 21.1458, 79.0882],
  ["Surat", 21.1702, 72.8311],
  ["Visakhapatnam", 17.6868, 83.2185],
  ["Coimbatore", 11.0168, 76.9558],
  ["Mysuru", 12.2958, 76.6394],
  ["Ranchi", 23.3441, 85.3096],
  ["Vadodara", 22.3072, 73.1812],
] as const;

const distanceSquared = (
  lat1: number,
  lon1: number,
  lat2: number,
  lon2: number,
) => {
  const latScale = 111.32;
  const lonScale = 111.32 * Math.cos((lat1 * Math.PI) / 180);
  const dLat = (lat2 - lat1) * latScale;
  const dLon = (lon2 - lon1) * lonScale;
  return dLat * dLat + dLon * dLon;
};

function App() {
  const [leadDay, setLeadDay] = useState(1);
  const [gridData, setGridData] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [isExplaining, setIsExplaining] = useState(false);
  const [selectedGrid, setSelectedGrid] = useState<any>(null);
  const [explainData, setExplainData] = useState<any>(null);
  const [isDrawerCollapsed, setIsDrawerCollapsed] = useState(false);
  const [mapSearch, setMapSearch] = useState("");
  const [focusPoint, setFocusPoint] = useState<{
    lat: number;
    lon: number;
  } | null>(null);

  const mapPanelRef = usePanelRef();
  const xaiPanelRef = usePanelRef();
  const drawerPanelRef = usePanelRef();

  // Fetch forecast data when leadDay changes (keeps current grid selected if possible)
  useEffect(() => {
    let isCurrent = true;
    const fetchData = async () => {
      setLoading(true);
      try {
        const res = await fetch("http://localhost:8000/api/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ lead_day: leadDay }),
        });
        const data = await res.json();
        const items = data.data || [];
        if (!isCurrent) return;

        setGridData(items);
        if (items.length > 0) {
          // Maintain existing selected location if still present, or pick highest risk
          setSelectedGrid((prev: any) => {
            if (prev) {
              const matched = items.find((d: any) => d.id === prev.id);
              if (matched) return matched;
            }
            const sorted = [...items].sort(
              (a: any, b: any) => b.bust_prob - a.bust_prob,
            );
            return sorted[0];
          });
        }
      } catch (err) {
        console.error("Failed to fetch grid data", err);
      } finally {
        if (isCurrent) setLoading(false);
      }
    };
    fetchData();
    return () => {
      isCurrent = false;
    };
  }, [leadDay]);

  // Fetch explainability point data when selectedGrid or leadDay updates
  useEffect(() => {
    if (!selectedGrid) return;
    const controller = new AbortController();
    setIsExplaining(true);

    const fetchExplain = async () => {
      try {
        const res = await fetch("http://localhost:8000/api/explain_point", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            lat: selectedGrid.lat,
            lon: selectedGrid.lon,
            lead_day: leadDay,
          }),
          signal: controller.signal,
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        setExplainData(data);
      } catch (err: any) {
        if (err.name !== "AbortError") {
          console.error("Failed to fetch explain data", err);
          // Fall back gracefully to selectedGrid data
          setExplainData({
            ...selectedGrid,
            weather_source: "fallback",
          });
        }
      } finally {
        if (!controller.signal.aborted) {
          setIsExplaining(false);
        }
      }
    };
    fetchExplain();
    return () => {
      controller.abort();
    };
  }, [selectedGrid?.id, leadDay]);

  const handleSelectGrid = useCallback((grid: any) => {
    setSelectedGrid(grid);
  }, []);

  const handleMapSearch = useCallback(async () => {
    const query = mapSearch.trim();
    if (!query || gridData.length === 0) return;

    const normalizedQuery = query.toLowerCase().replace(/[^a-z0-9]+/g, "");
    const gridMatch = gridData.find((grid: any) => {
      const gridLabel = `${grid.district_name || ""} ${grid.state_name || ""}`
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "");
      return (
        gridLabel.includes(normalizedQuery) ||
        normalizedQuery.includes(gridLabel)
      );
    });

    let lat: number;
    let lon: number;
    let displayName = query;
    if (gridMatch) {
      setSelectedGrid(gridMatch);
      setFocusPoint({ lat: gridMatch.lat, lon: gridMatch.lon });
      setMapSearch(`${gridMatch.district_name}, ${gridMatch.state_name}`);
      return;
    }

    const city = CITY_LOCATIONS.find(([name]) => {
      const normalizedName = name.toLowerCase().replace(/[^a-z0-9]+/g, "");
      return (
        normalizedName === normalizedQuery ||
        normalizedName.includes(normalizedQuery) ||
        normalizedQuery.includes(normalizedName)
      );
    });

    if (city) {
      [, lat, lon] = city;
      displayName = city[0];
    } else {
      try {
        const response = await fetch(
          `https://nominatim.openstreetmap.org/search?format=jsonv2&limit=1&countrycodes=in&q=${encodeURIComponent(query)}`,
        );
        const results = await response.json();
        if (!results?.[0]) return;
        lat = Number(results[0].lat);
        lon = Number(results[0].lon);
        if (!Number.isFinite(lat) || !Number.isFinite(lon)) return;
        displayName =
          results[0].display_name?.split(",").slice(0, 2).join(",") || query;
      } catch (error) {
        console.error("Location search failed", error);
        return;
      }
    }

    const nearestGrid = gridData.reduce((nearest, grid) => {
      if (!nearest) return grid;
      return distanceSquared(lat, lon, grid.lat, grid.lon) <
        distanceSquared(lat, lon, nearest.lat, nearest.lon)
        ? grid
        : nearest;
    }, null as any);

    if (nearestGrid) {
      setSelectedGrid(nearestGrid);
      setFocusPoint({ lat: nearestGrid.lat, lon: nearestGrid.lon });
      setMapSearch(displayName);
    }
  }, [gridData, mapSearch]);

  // Quick-toggle collapse & expand helpers
  const handleExpandMap = () => {
    if (mapPanelRef.current && xaiPanelRef.current) {
      mapPanelRef.current.resize("88%");
      xaiPanelRef.current.resize("12%");
    }
  };

  const handleExpandXAI = () => {
    if (mapPanelRef.current && xaiPanelRef.current) {
      mapPanelRef.current.resize("30%");
      xaiPanelRef.current.resize("70%");
    }
  };

  const handleResetHorizontal = () => {
    if (mapPanelRef.current && xaiPanelRef.current) {
      mapPanelRef.current.resize("60%");
      xaiPanelRef.current.resize("40%");
    }
  };

  const handleToggleDrawer = () => {
    if (drawerPanelRef.current) {
      if (isDrawerCollapsed) {
        drawerPanelRef.current.resize("28%");
        setIsDrawerCollapsed(false);
      } else {
        drawerPanelRef.current.resize("6%");
        setIsDrawerCollapsed(true);
      }
    }
  };

  const highRiskCount = gridData.filter((d) => d.bust_prob > 0.5).length;
  const maxBust = gridData.length
    ? Math.max(...gridData.map((d) => d.bust_prob))
    : 0;
  const maxBustPct = maxBust * 100;
  const modelConfidencePct = 100 - maxBustPct;

  const sortedByRisk = [...gridData].sort((a, b) => b.bust_prob - a.bust_prob);
  const top12Risks = sortedByRisk.slice(0, 12);

  const isExplainMatching =
    explainData &&
    selectedGrid &&
    Math.abs(explainData.lat - selectedGrid.lat) < 0.01 &&
    Math.abs(explainData.lon - selectedGrid.lon) < 0.01;
  const activeGrid = isExplainMatching
    ? { ...selectedGrid, ...explainData }
    : selectedGrid || (sortedByRisk.length > 0 ? sortedByRisk[0] : null);

  return (
    <div className="h-screen w-screen flex flex-col overflow-hidden bg-background">
      {/* Top Bar Header - Fixed dimensions to eliminate horizontal layout jitter */}
      <header className="h-20 bg-surface/90 backdrop-blur-lg border-b border-white/10 flex items-center justify-between px-8 z-10 shrink-0 select-none">
        <div className="flex items-center space-x-4 shrink-0">
          <div className="w-10 h-10 rounded-lg bg-accentSafeCyan/20 flex items-center justify-center shadow-inner shrink-0">
            <Zap className="text-accentSafeCyan w-6 h-6" />
          </div>
          <div>
            <h1 className="font-bold text-xl tracking-wide text-white whitespace-nowrap">
              MoES BUST-DETECT
            </h1>
            <p className="text-slate-400 text-xs tracking-wider whitespace-nowrap">
              NCMRWF Operational Forecast Warning System
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-8 shrink-0">
          {/* Lead Day Slider */}
          <div className="flex items-center space-x-3 shrink-0">
            <span className="text-sm text-slate-300 font-semibold uppercase tracking-wider whitespace-nowrap">
              Lead Day
            </span>
            <input
              type="range"
              min="1"
              max="10"
              value={leadDay}
              onChange={(e) => setLeadDay(Number(e.target.value))}
              className="w-36 accent-accentSafeCyan cursor-pointer shrink-0"
            />
            <span className="font-mono text-accentSafeCyan font-bold text-xl w-10 text-left shrink-0 tabular-nums">
              +{leadDay}
            </span>
          </div>

          <div className="h-10 w-px bg-white/10 shrink-0"></div>

          {/* Metric Badges with Tabular Numerals & Static Widths */}
          <div className="flex space-x-6 shrink-0">
            <div className="flex flex-col items-center min-w-[70px]">
              <span className="text-slate-400 text-xs font-semibold uppercase tracking-wider whitespace-nowrap">
                Active Grids
              </span>
              <span className="text-white font-bold text-lg font-mono tabular-nums">
                {gridData.length}
              </span>
            </div>
            <div className="flex flex-col items-center min-w-[65px]">
              <span className="text-slate-400 text-xs font-semibold uppercase tracking-wider whitespace-nowrap">
                High Risk
              </span>
              <span
                className={`font-bold text-lg font-mono tabular-nums ${highRiskCount > 0 ? "text-accentCritical" : "text-accentSafe"}`}
              >
                {highRiskCount}
              </span>
            </div>
            <div
              className={`flex items-center space-x-2 px-3.5 py-1.5 rounded-lg border shadow-sm ${
                maxBust <= 0.3
                  ? "bg-emerald-500/10 border-emerald-500/20"
                  : maxBust <= 0.5
                    ? "bg-amber-500/10 border-amber-500/20"
                    : "bg-red-500/10 border-red-500/20"
              }`}
            >
              <span className="text-xs font-bold text-slate-200 uppercase tracking-wider whitespace-nowrap">
                NWP CONFIDENCE:
              </span>
              <span
                className={`font-bold text-sm font-mono tabular-nums ${maxBust <= 0.3 ? "text-emerald-400" : maxBust <= 0.5 ? "text-amber-400" : "text-red-400"}`}
              >
                {modelConfidencePct.toFixed(1)}%
              </span>
              <span className="text-slate-500">|</span>
              <span className="text-xs font-bold text-slate-200 uppercase tracking-wider whitespace-nowrap">
                MAX BUST:
              </span>
              <span
                className={`font-bold text-sm font-mono tabular-nums ${maxBust > 0.5 ? "text-red-400" : maxBust > 0.3 ? "text-amber-400" : "text-emerald-400"}`}
              >
                {maxBustPct.toFixed(1)}%
              </span>
            </div>
          </div>

          <div className="h-10 w-px bg-white/10 shrink-0"></div>

          {/* Fixed-width inference indicator */}
          <div className="flex items-center space-x-2 bg-accentSafe/10 px-3.5 py-1.5 rounded-lg border border-accentSafe/20 shadow-sm w-44 justify-center shrink-0">
            <Server
              className={`w-4 h-4 shrink-0 ${loading || isExplaining ? "animate-spin text-accentSafeCyan" : "text-accentSafe"}`}
            />
            <span className="text-[13px] font-semibold text-accentSafe whitespace-nowrap">
              {loading
                ? "COMPUTING GRID..."
                : isExplaining
                  ? "ANALYZING POINT..."
                  : "INFERENCE ACTIVE"}
            </span>
          </div>
        </div>
      </header>

      {/* Main Resizable Layout */}
      <main className="flex-1 overflow-hidden p-3 relative flex flex-col">
        {/* Vertical Panel Group: Top (Map + XAI) vs Bottom (Top 10 Drawer) */}
        <PanelGroup orientation="vertical" className="flex-1 w-full h-full">
          {/* Top Panel: Horizontal Split between 3D Map and XAI Panel */}
          <Panel
            defaultSize="75%"
            minSize="35%"
            className="flex flex-col overflow-hidden pb-1"
          >
            <PanelGroup orientation="horizontal" className="h-full w-full">
              {/* Left Panel: 3D Map Component */}
              <Panel
                panelRef={mapPanelRef}
                defaultSize="60%"
                minSize="20%"
                maxSize="90%"
                className="relative flex flex-col h-full rounded-xl overflow-hidden border border-white/10 bg-surface/50"
              >
                {/* High-Risk Alert Banner (Positioned statically on extreme top-left of map) */}
                {highRiskCount > 0 && (
                  <div className="absolute top-3 left-3 z-30 flex items-center space-x-2.5 bg-red-950/85 backdrop-blur-md border border-red-500/60 px-3.5 py-1.5 rounded-lg shadow-2xl pointer-events-none select-none">
                    <AlertTriangle className="text-red-400 w-4.5 h-4.5 shrink-0 animate-pulse" />
                    <span className="text-red-200 text-xs sm:text-sm font-bold uppercase tracking-wider">
                      {highRiskCount} High Risk{" "}
                      {highRiskCount === 1 ? "Region" : "Regions"} Detected
                    </span>
                  </div>
                )}

                {/* Quick-Toggle Toolbar (Positioned on extreme top-right of map) */}
                <div className="absolute top-3 right-3 z-30 flex items-center space-x-1.5 bg-slate-900/85 backdrop-blur-md border border-slate-700/70 rounded-lg p-1 shadow-xl select-none">
                  <button
                    onClick={handleExpandMap}
                    title="Expand 3D Map view to 88%"
                    className="flex items-center space-x-1 px-2.5 py-1 text-xs font-semibold text-slate-200 hover:text-white hover:bg-slate-700/70 rounded transition-all"
                  >
                    <Maximize2 className="w-3.5 h-3.5 text-cyan-400" />
                    <span>Expand Map ⛶</span>
                  </button>
                  <button
                    onClick={handleResetHorizontal}
                    title="Reset to 60/40 Split"
                    className="px-2 py-1 text-xs font-semibold text-slate-300 hover:text-white hover:bg-slate-700/70 rounded transition-all"
                  >
                    60/40
                  </button>
                  <button
                    onClick={handleExpandXAI}
                    title="Expand Explainable AI panel"
                    className="flex items-center space-x-1 px-2.5 py-1 text-xs font-semibold text-slate-200 hover:text-white hover:bg-slate-700/70 rounded transition-all"
                  >
                    <Sliders className="w-3.5 h-3.5 text-amber-400" />
                    <span>Expand XAI ⛶</span>
                  </button>
                </div>

                <MapComponent
                  data={gridData}
                  selectedGridId={selectedGrid?.id || null}
                  onSelectGrid={handleSelectGrid}
                  focusPoint={focusPoint}
                />
                <form
                  onSubmit={(event) => {
                    event.preventDefault();
                    handleMapSearch();
                  }}
                  className="absolute top-3 left-3 z-30 flex w-[min(380px,calc(100%-1.5rem))] items-center rounded-lg border border-white/15 bg-slate-900/90 p-2 shadow-xl backdrop-blur-md"
                >
                  <Search className="ml-2 mr-2 h-5 w-5 shrink-0 text-cyan-300" />
                  <input
                    value={mapSearch}
                    onChange={(event) => setMapSearch(event.target.value)}
                    list="india-map-locations"
                    placeholder="Search city or district"
                    aria-label="Search city or district"
                    className="min-w-0 flex-1 bg-transparent px-1.5 py-2 text-base font-medium text-white outline-none placeholder:text-slate-400"
                  />
                  <datalist id="india-map-locations">
                    {CITY_LOCATIONS.map(([name]) => (
                      <option key={name} value={name} />
                    ))}
                  </datalist>
                  <button
                    type="submit"
                    className="rounded-md bg-cyan-500/20 px-4 py-2 text-sm font-bold text-cyan-200 transition hover:bg-cyan-500/35"
                  >
                    Go
                  </button>
                </form>
              </Panel>

              {/* Draggable Vertical Splitter Handle */}
              <PanelResizeHandle
                className="relative w-3 mx-0.5 flex items-center justify-center cursor-col-resize group z-20 select-none hover:bg-cyan-500/20 active:bg-cyan-500/40 transition-colors rounded-sm"
                title="Drag left/right to resize 3D Map vs XAI Panel"
              >
                <div className="w-1.5 h-12 rounded-full bg-slate-700 group-hover:bg-cyan-400 group-active:bg-cyan-300 transition-colors flex flex-col items-center justify-center space-y-1 shadow-sm">
                  <div className="w-0.5 h-0.5 rounded-full bg-slate-300" />
                  <div className="w-0.5 h-0.5 rounded-full bg-slate-300" />
                  <div className="w-0.5 h-0.5 rounded-full bg-slate-300" />
                </div>
              </PanelResizeHandle>

              {/* Right Panel: XAI Dashboard */}
              <Panel
                panelRef={xaiPanelRef}
                defaultSize="40%"
                minSize="10%"
                maxSize="80%"
                className="relative h-full bg-surface/80 backdrop-blur-md rounded-xl p-5 overflow-y-auto border border-white/10"
              >
                <XAIDashboard
                  activeGrid={activeGrid}
                  leadDay={leadDay}
                  onSelectLeadDay={setLeadDay}
                  isExplaining={isExplaining}
                />
              </Panel>
            </PanelGroup>
          </Panel>

          {/* Draggable Horizontal Splitter Handle */}
          <PanelResizeHandle
            className="relative h-3.5 my-0.5 w-full flex items-center justify-center cursor-row-resize group z-20 select-none hover:bg-cyan-500/20 active:bg-cyan-500/40 transition-colors rounded-sm"
            title="Drag up/down to resize Top 10 High-Risk table"
          >
            <div className="h-1.5 w-16 rounded-full bg-slate-700 group-hover:bg-cyan-400 group-active:bg-cyan-300 transition-colors flex items-center justify-center space-x-1 shadow-sm">
              <div className="w-0.5 h-0.5 rounded-full bg-slate-300" />
              <div className="w-0.5 h-0.5 rounded-full bg-slate-300" />
              <div className="w-0.5 h-0.5 rounded-full bg-slate-300" />
            </div>
          </PanelResizeHandle>

          {/* Bottom Panel: Top 10 High-Risk Drawer (Persistent, non-rebuilding DOM) */}
          <Panel
            panelRef={drawerPanelRef}
            defaultSize="25%"
            minSize="6%"
            maxSize="65%"
            className="pt-0.5 flex flex-col overflow-hidden"
          >
            <div className="h-full bg-surface/90 backdrop-blur-md rounded-xl border border-white/10 p-3.5 flex flex-col overflow-hidden shadow-xl">
              {/* Persistent Header */}
              <div className="flex items-center justify-between shrink-0 pb-2.5 border-b border-white/10">
                <div className="flex items-center space-x-3 truncate">
                  <AlertTriangle className="w-5 h-5 text-accentCritical shrink-0" />
                  <span className="text-[15px] font-extrabold text-accentCritical uppercase tracking-wider whitespace-nowrap">
                    Top 12 High-Risk Failure Zones
                  </span>
                  <span className="text-xs bg-red-500/20 text-red-300 border border-red-500/30 px-2.5 py-0.5 rounded-full font-bold shrink-0">
                    {top12Risks.length} Grids
                  </span>
                </div>
                <div className="flex items-center space-x-2 shrink-0">
                  <button
                    onClick={handleToggleDrawer}
                    className="text-xs font-semibold text-slate-300 hover:text-white bg-slate-800 hover:bg-slate-700 border border-slate-600 px-2.5 py-1 rounded flex items-center space-x-1 transition-all"
                  >
                    {isDrawerCollapsed ? (
                      <>
                        <ChevronUp className="w-3.5 h-3.5 text-cyan-400" />
                        <span>Expand Table</span>
                      </>
                    ) : (
                      <>
                        <ChevronDown className="w-3.5 h-3.5 text-slate-400" />
                        <span>Collapse Ribbon</span>
                      </>
                    )}
                  </button>
                </div>
              </div>

              {/* Scrollable Risk Grid - Always rendered, naturally cropped when collapsed */}
              <div className="flex-1 overflow-y-auto pt-3 pr-1">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2.5">
                  {top12Risks.map((risk, idx) => {
                    const isCritical = risk.bust_prob > 0.5;
                    const isWatch =
                      risk.bust_prob > 0.3 && risk.bust_prob <= 0.5;
                    const isSelected = selectedGrid?.id === risk.id;
                    return (
                      <div
                        key={idx}
                        className={`border px-3.5 py-2.5 rounded-xl flex justify-between items-center cursor-pointer transition-all ${
                          isSelected
                            ? "ring-2 ring-cyan-400 bg-slate-800/95 border-cyan-400 shadow-lg"
                            : isCritical
                              ? "bg-red-500/10 border-red-500/30 hover:bg-red-500/20 hover:border-red-500/50"
                              : isWatch
                                ? "bg-amber-500/10 border-amber-500/30 hover:bg-amber-500/20 hover:border-amber-500/50"
                                : "bg-emerald-500/10 border-emerald-500/30 hover:bg-emerald-500/20 hover:border-emerald-500/50"
                        }`}
                        onClick={() => handleSelectGrid(risk)}
                      >
                        <div className="flex flex-col truncate pr-3">
                          <div className="flex items-center space-x-2 truncate">
                            <span className="text-[13.5px] font-extrabold text-slate-400">
                              #{idx + 1}
                            </span>
                            <span className="text-[15px] font-bold text-slate-100 truncate">
                              {risk.district_name}, {risk.state_name}
                            </span>
                          </div>
                          <div className="flex items-center space-x-2 text-[12.5px] text-slate-300 font-medium mt-1">
                            <span>
                              {risk.lat.toFixed(1)}°N, {risk.lon.toFixed(1)}°E
                            </span>
                            {risk.primary_driver && (
                              <>
                                <span className="text-slate-500">·</span>
                                <span className="text-cyan-400 font-semibold truncate">
                                  {risk.primary_driver}
                                </span>
                              </>
                            )}
                          </div>
                        </div>
                        <div className="flex items-center justify-end gap-3 shrink-0 pl-3 min-w-[190px]">
                          <span
                            className={`text-[19px] md:text-[20px] font-extrabold font-mono whitespace-nowrap ${isCritical ? "text-red-400" : isWatch ? "text-amber-400" : "text-emerald-400"}`}
                          >
                            {(risk.bust_prob * 100).toFixed(1)}% Risk
                          </span>
                          <span
                            className={`text-[17px] md:text-[18px] font-extrabold font-mono whitespace-nowrap ${isCritical ? "text-red-300" : isWatch ? "text-amber-300" : "text-emerald-300"}`}
                          >
                            {(100 - risk.bust_prob * 100).toFixed(1)}% NWP
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          </Panel>
        </PanelGroup>
      </main>
    </div>
  );
}

export default App;
