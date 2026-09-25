import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import {
  BrainCircuit,
  History,
  ShieldAlert,
  MapPin,
  CheckCircle,
  AlertTriangle,
  Thermometer,
  Droplets,
  Wind,
  Gauge,
  CloudRain,
  Calendar,
  CloudSun,
  CloudDrizzle,
  CloudLightning,
  Sun,
} from "lucide-react";

interface XAIProps {
  activeGrid: any;
  leadDay: number;
  onSelectLeadDay?: (day: number) => void;
  isExplaining?: boolean;
}

// Helper to highlight key variables (locations, meteorological values, LIME weights, drivers)
function escapeRegex(str: string) {
  return str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function highlightInsightText(
  rawText: string,
  district?: string,
  state?: string,
): React.ReactNode[] {
  const specialTokens: string[] = [];
  if (district && district !== "Unknown")
    specialTokens.push(escapeRegex(district));
  if (state && state !== "Unknown") specialTokens.push(escapeRegex(state));

  const locationPattern =
    specialTokens.length > 0 ? `(?:${specialTokens.join("|")})|` : "";
  const pattern = new RegExp(
    `(${locationPattern}[-+]?\\d+(?:\\.\\d+)?\\s*(?:°C|mm²|m\\/s|%|hPa)|(?:local weight:\\s*[-+]?\\d+(?:\\.\\d+)?)|(?:local surrogate weight:\\s*[-+]?\\d+(?:\\.\\d+)?)|[-+]\\d+\\.\\d{3,}|\\b(?:Rainfall Acceleration|Rain Acceleration|Temperature Anomaly|Temp Anomaly|Vertical Wind Shear|Wind Shear|Humidity Pooling|Humidity|Surface Pressure Drop|Pressure Drop|LIME|SHAP)\\b)`,
    "gi",
  );

  const parts: React.ReactNode[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = pattern.exec(rawText)) !== null) {
    if (match.index > lastIndex) {
      parts.push(rawText.substring(lastIndex, match.index));
    }
    const token = match[0];
    // Numbers with weather units -> bold yellow (amber-400)
    if (/(?:°C|mm²|m\/s|%|hPa)/.test(token)) {
      parts.push(
        <span key={match.index} className="text-amber-400 font-semibold">
          {token}
        </span>,
      );
    } else {
      // Locations, LIME weights, drivers -> bold cyan (cyan-400)
      parts.push(
        <span key={match.index} className="text-cyan-400 font-semibold">
          {token}
        </span>,
      );
    }
    lastIndex = pattern.lastIndex;
  }

  if (lastIndex < rawText.length) {
    parts.push(rawText.substring(lastIndex));
  }

  return parts;
}

// Custom High-Contrast Tooltip for Recharts
const CustomTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0];
    return (
      <div className="bg-[#1E293B] border border-[#475569] rounded-lg p-3 shadow-2xl backdrop-blur-md pointer-events-none">
        <p className="text-[#FFFFFF] text-[14px] font-bold tracking-wide mb-1">
          {data.payload.name}
        </p>
        <div className="flex items-center space-x-2 text-[14px]">
          <span className="text-slate-300 font-medium">SHAP Impact:</span>
          <span className="text-[#38BDF8] font-bold font-mono">
            {Number(data.value).toFixed(4)}
          </span>
        </div>
      </div>
    );
  }
  return null;
};

function getWeatherIcon(code: number) {
  if (code === 0 || code === 1)
    return <Sun className="w-4 h-4 text-amber-400 shrink-0" />;
  if (code === 2 || code === 3)
    return <CloudSun className="w-4 h-4 text-slate-300 shrink-0" />;
  if (code >= 51 && code <= 57)
    return <CloudDrizzle className="w-4 h-4 text-blue-300 shrink-0" />;
  if (code >= 61 && code <= 67)
    return <CloudRain className="w-4 h-4 text-blue-400 shrink-0" />;
  if (code >= 80 && code <= 82)
    return <CloudRain className="w-4 h-4 text-cyan-400 shrink-0" />;
  if (code >= 95)
    return <CloudLightning className="w-4 h-4 text-purple-400 shrink-0" />;
  return <CloudSun className="w-4 h-4 text-slate-300 shrink-0" />;
}

export default function XAIDashboard({
  activeGrid,
  leadDay,
  onSelectLeadDay,
  isExplaining,
}: XAIProps) {
  if (!activeGrid) {
    return (
      <div className="h-full flex flex-col items-center justify-center text-slate-400 space-y-3">
        <MapPin className="w-12 h-12 text-slate-500" />
        <p className="text-lg font-semibold">Click a grid column to analyze</p>
        <p className="text-sm text-slate-500">
          Select any 3D column on the map for XAI insights
        </p>
      </div>
    );
  }

  // ─── SHAP bar chart data ────────────────────────────────────────────
  // Use real SHAP values from backend if available; otherwise compute locally
  const shapData = activeGrid.shap_values
    ? activeGrid.shap_values.map((sv: any) => ({
        name: sv.feature,
        val: Math.abs(sv.value),
      }))
    : [
        { name: "Rain Acceleration", val: activeGrid.rain_squared || 0 },
        { name: "Temp Anomaly", val: activeGrid.temp_anomaly || 0 },
        { name: "Wind Shear", val: (activeGrid.wind_shear || 0) / 2 },
        { name: "Humidity", val: (activeGrid.humidity || 0) / 8 },
        {
          name: "Pressure Drop",
          val:
            activeGrid.pressure_drop ||
            (activeGrid.pressure ? Math.abs(1013 - activeGrid.pressure) : 0),
        },
      ].sort((a, b) => b.val - a.val);

  // ─── LIME data ──────────────────────────────────────────────────────
  const limeWeights = activeGrid.lime_weights || null;
  const limeAgreement = activeGrid.lime_agreement ?? null;

  // ─── Scalar values ──────────────────────────────────────────────────
  const prob = activeGrid.bust_prob || 0;
  const nwpConfidencePct = Math.max(0, Math.min(100, 100 - prob * 100));
  const district = activeGrid.district_name || "Unknown";
  const state = activeGrid.state_name || "Unknown";
  const openMeteo10Day = activeGrid.openmeteo_10day || null;
  const weatherSource = activeGrid.weather_source || "deterministic";

  // ─── Feature values ─────────────────────────────────────────────────
  const featureValues = activeGrid.feature_values || {
    temp_anomaly: activeGrid.temp_anomaly || 0,
    rain_accel: activeGrid.rain_squared || 0,
    wind_shear: activeGrid.wind_shear || 0,
    humidity: activeGrid.humidity || 0,
    pressure_drop:
      activeGrid.pressure_drop ||
      (activeGrid.pressure ? Math.abs(1013 - activeGrid.pressure) : 0),
  };

  // ─── Dynamic insight from backend ───────────────────────────────────
  const dynamicInsight = activeGrid.dynamic_insight || null;

  // ─── Synoptic match ─────────────────────────────────────────────────
  const synoptic = activeGrid.synoptic_match || null;
  const similarityScore = synoptic
    ? synoptic.similarity?.toFixed(1)
    : activeGrid.similarity_score?.toFixed(1) || "—";
  const eventMatch = synoptic ? synoptic.event : activeGrid.past_event || "N/A";

  // ─── LIME ordered data (Left = Primary Driver, Right = Secondary Driver) ───
  const primaryDriverName =
    activeGrid.primary_driver ||
    (shapData.length > 0 ? shapData[0].name : "Temp Anomaly");
  const secondaryDriverName =
    activeGrid.secondary_driver && activeGrid.secondary_driver !== "Stable"
      ? activeGrid.secondary_driver
      : shapData.length > 1
        ? shapData[1].name
        : limeWeights?.[1]?.feature || "Wind Shear";

  const limeWeightMap: Record<string, number> = {};
  if (limeWeights && Array.isArray(limeWeights)) {
    limeWeights.forEach((lw: any) => {
      limeWeightMap[lw.feature] = lw.weight;
    });
  }

  const primaryWeight =
    limeWeightMap[primaryDriverName] ?? (limeWeights?.[0]?.weight || 0.0);
  const secondaryWeight =
    limeWeightMap[secondaryDriverName] ?? (limeWeights?.[1]?.weight || 0.0);

  const orderedLimeCards = [
    { feature: primaryDriverName, weight: primaryWeight, role: "Primary" },
    {
      feature: secondaryDriverName,
      weight: secondaryWeight,
      role: "Secondary",
    },
  ];

  const probColor = prob > 0.5 ? "#EF4444" : prob > 0.3 ? "#F59E0B" : "#10B981";

  // ─── Feature value display items ────────────────────────────────────
  const featureItems = [
    {
      label: "Temp Anomaly",
      value: `${featureValues.temp_anomaly}°C`,
      icon: Thermometer,
      color: "text-orange-400",
    },
    {
      label: "Rain Accel",
      value: `${featureValues.rain_accel} mm²`,
      icon: CloudRain,
      color: "text-blue-400",
    },
    {
      label: "Wind Shear",
      value: `${featureValues.wind_shear} m/s`,
      icon: Wind,
      color: "text-emerald-400",
    },
    {
      label: "Humidity",
      value: `${featureValues.humidity}%`,
      icon: Droplets,
      color: "text-cyan-400",
    },
    {
      label: "Pressure Drop",
      value: `${featureValues.pressure_drop} hPa`,
      icon: Gauge,
      color: "text-purple-400",
    },
  ];

  return (
    <div className="flex flex-col h-full space-y-4 font-sans">
      {/* ═══════ Header ═══════ */}
      <div className="border-b border-white/10 pb-4">
        <h2 className="text-2xl font-bold text-white flex items-center tracking-tight">
          <BrainCircuit className="mr-3 w-7 h-7 text-accentSafeCyan" />
          Explainable AI Panel
        </h2>
        <p className="text-base text-slate-200 font-medium mt-2">
          📍 {district}, {state}
        </p>
        <p className="text-sm text-slate-400 mt-1">
          {activeGrid.lat?.toFixed(2)}°N, {activeGrid.lon?.toFixed(2)}°E
          &nbsp;·&nbsp; Lead Day +{leadDay}
        </p>
        <div className="mt-3 flex items-center justify-between">
          <div className="flex items-end gap-5">
            <div className="flex items-baseline gap-2">
              <span
                className="text-4xl font-bold font-mono"
                style={{ color: probColor }}
              >
                {(prob * 100).toFixed(1)}%
              </span>
              <span className="text-lg text-slate-400 font-medium">Risk</span>
            </div>
            <div className="flex items-baseline gap-2 pb-0.5">
              <span className="text-3xl font-bold font-mono text-emerald-300">
                {nwpConfidencePct.toFixed(1)}%
              </span>
              <span className="text-sm font-semibold text-emerald-200 whitespace-nowrap">
                NWP Confidence
              </span>
            </div>
          </div>

          <div className="flex items-center space-x-2 bg-slate-800/80 border border-slate-700/70 px-3 py-1.5 rounded-lg shadow-sm">
            <span
              className={`w-2 h-2 rounded-full ${isExplaining ? "bg-cyan-400 animate-ping" : weatherSource === "openmeteo" ? "bg-emerald-400 animate-pulse" : "bg-cyan-400"}`}
            />
            <span className="text-xs font-semibold text-slate-200">
              {isExplaining
                ? "Analyzing Point..."
                : weatherSource === "openmeteo"
                  ? "Open-Meteo 10-Day Feed"
                  : "Deterministic Model"}
            </span>
          </div>
        </div>
      </div>

      {/* ═══════ Parameter Metrics Ribbon (5 Cards) ═══════ */}
      <div className="grid grid-cols-5 gap-2.5">
        {featureItems.map((fi) => (
          <div
            key={fi.label}
            className="border border-slate-700/60 bg-slate-800/70 rounded-lg p-2.5 flex flex-col items-center text-center shadow-sm"
          >
            <fi.icon className={`w-4.5 h-4.5 ${fi.color} mb-1.5`} />
            <span className="text-[12px] font-bold text-slate-300 uppercase tracking-wider leading-tight">
              {fi.label}
            </span>
            <span className="text-[16px] md:text-[17px] font-extrabold text-white mt-1 font-mono">
              {fi.value}
            </span>
          </div>
        ))}
      </div>

      {/* ═══════ 10-Day Open-Meteo Future Forecast & Risk Outlook ═══════ */}
      {openMeteo10Day && openMeteo10Day.length > 0 ? (
        <div className="border border-slate-700/50 bg-slate-800/40 rounded-xl p-4">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center space-x-2">
              <Calendar className="w-4.5 h-4.5 text-cyan-400" />
              <h3 className="text-sm font-bold text-slate-200 uppercase tracking-wider">
                Open-Meteo 10-Day Future Outlook
              </h3>
            </div>
            <span className="text-xs text-slate-400">
              Click any day to jump to that lead day
            </span>
          </div>

          <div className="flex space-x-2.5 overflow-x-auto pb-2 scrollbar-thin scrollbar-thumb-slate-700 scrollbar-track-transparent">
            {openMeteo10Day.map((day: any) => {
              const isSelected = day.lead_day === leadDay;
              const bustPct = day.bust_prob * 100;
              const confPct = Math.max(
                0,
                Math.min(100, Number(day.confidence_pct ?? 100 - bustPct)),
              );
              const confStyle =
                confPct >= 70
                  ? "bg-emerald-500/20 text-emerald-200 border-emerald-400/50"
                  : confPct >= 50
                    ? "bg-amber-500/20 text-amber-200 border-amber-400/50"
                    : "bg-red-500/20 text-red-200 border-red-400/50";

              return (
                <div
                  key={day.lead_day}
                  onClick={() => onSelectLeadDay?.(day.lead_day)}
                  className={`flex h-[146px] flex-col min-w-[158px] max-w-[158px] p-2.5 rounded-lg border cursor-pointer transition-all shrink-0 ${
                    isSelected
                      ? "border-cyan-400 bg-slate-700/90 shadow-lg ring-1 ring-cyan-400"
                      : "border-slate-700/60 bg-slate-800/70 hover:bg-slate-700/50 hover:border-slate-500"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span
                      className={`text-[14px] font-bold font-mono ${isSelected ? "text-cyan-300 font-extrabold" : "text-slate-400"}`}
                    >
                      Day +{day.lead_day}
                    </span>
                    {isSelected && (
                      <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
                    )}
                  </div>

                  <span className="text-[12px] text-slate-400 mt-0.5">
                    {day.date.slice(5)}
                  </span>

                  <div className="flex items-center space-x-1.5 mt-2">
                    {getWeatherIcon(day.weather_code)}
                    <span
                      className="text-[14px] font-medium text-slate-200 truncate"
                      title={day.weather_desc}
                    >
                      {day.weather_desc}
                    </span>
                  </div>

                  <div
                    className={`mt-auto w-full text-center py-1.5 px-1 rounded text-[13px] font-mono font-bold border ${confStyle}`}
                  >
                    NWP Conf: {confPct.toFixed(1)}%
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      ) : isExplaining ? (
        <div className="border border-slate-700/50 bg-slate-800/30 rounded-xl p-4 flex items-center justify-center space-x-3 text-slate-400 animate-pulse">
          <div className="w-4 h-4 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin" />
          <span className="text-xs font-medium text-slate-300">
            Fetching 10-day Open-Meteo forecast...
          </span>
        </div>
      ) : null}

      {/* ═══════ SHAP Feature Impact Chart ═══════ */}
      <div
        className="border border-slate-700/50 bg-slate-800/40 rounded-xl p-5"
        style={{ minHeight: "270px" }}
      >
        <h3 className="text-sm font-bold text-slate-300 uppercase mb-4 tracking-wider">
          XAI Drivers of NWP Model Instability
        </h3>
        <ResponsiveContainer width="100%" height={210}>
          <BarChart
            data={shapData}
            layout="vertical"
            margin={{ top: 0, right: 20, left: 10, bottom: 0 }}
          >
            <XAxis type="number" hide />
            <YAxis
              dataKey="name"
              type="category"
              axisLine={false}
              tickLine={false}
              tick={{
                fill: "#F1F5F9",
                fontSize: 13.5,
                fontWeight: 600,
                fontFamily: "inherit",
              }}
              width={150}
            />
            <Tooltip
              cursor={{ fill: "rgba(255,255,255,0.06)" }}
              content={<CustomTooltip />}
            />
            <Bar dataKey="val" radius={[0, 6, 6, 0]} barSize={22}>
              {shapData.map((_entry: any, index: number) => (
                <Cell
                  key={`cell-${index}`}
                  fill={
                    index === 0
                      ? "#EF4444"
                      : index === 1
                        ? "#F59E0B"
                        : "#06B6D4"
                  }
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* ═══════ LIME Local Surrogate Verification ═══════ */}
      <div className="border border-slate-700/50 bg-slate-800/40 rounded-xl p-4">
        <h3 className="text-sm font-bold text-slate-300 uppercase mb-3 tracking-wider">
          LIME Local Surrogate Verification
        </h3>

        {/* Agreement badge */}
        <div className="flex items-center space-x-2 mb-3">
          {limeAgreement !== null ? (
            limeAgreement ? (
              <div className="flex items-center space-x-2 bg-emerald-500/15 border border-emerald-500/30 px-3.5 py-1.5 rounded-full">
                <CheckCircle className="w-4 h-4 text-emerald-400" />
                <span className="text-[14px] font-semibold text-emerald-400">
                  SHAP–LIME Agreement
                </span>
              </div>
            ) : (
              <div className="flex items-center space-x-2 bg-amber-500/15 border border-amber-500/30 px-3.5 py-1.5 rounded-full">
                <AlertTriangle className="w-4 h-4 text-amber-400" />
                <span className="text-[14px] font-semibold text-amber-400">
                  Ranking Divergence
                </span>
              </div>
            )
          ) : (
            <div className="flex items-center space-x-2 bg-slate-500/15 border border-slate-500/30 px-3.5 py-1.5 rounded-full">
              <span className="text-[14px] font-semibold text-slate-300">
                Awaiting LIME data...
              </span>
            </div>
          )}
        </div>

        {/* LIME top-2 weights table: Left = Primary Driver, Right = Secondary Driver */}
        {limeWeights && limeWeights.length > 0 && (
          <div className="grid grid-cols-2 gap-2.5">
            {orderedLimeCards.map((lw, idx) => {
              const isNegative = lw.weight < 0;
              const weightColor = isNegative
                ? "text-red-400"
                : "text-emerald-400";
              const sign = lw.weight > 0 ? "+" : "";
              return (
                <div
                  key={idx}
                  className="bg-slate-800/80 border border-slate-700/60 rounded-lg px-3.5 py-2.5 flex justify-between items-center shadow-sm"
                >
                  <span className="text-[14.5px] font-bold text-slate-100 truncate pr-1">
                    {lw.feature}
                  </span>
                  <span
                    className={`text-[15px] font-extrabold font-mono ${weightColor}`}
                  >
                    {sign}
                    {lw.weight.toFixed(4)}
                  </span>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* ═══════ Operational Insights (Dynamic) ═══════ */}
      <div className="border border-slate-700/50 bg-slate-800/40 rounded-xl p-5">
        <h3 className="text-sm font-bold text-slate-300 uppercase mb-3 flex items-center tracking-wider">
          <ShieldAlert className="w-4.5 h-4.5 mr-2 text-amber-400" />
          Operational Insights
        </h3>
        <div
          className="text-[15.5px] leading-relaxed text-slate-100 font-sans border-l-4 border-cyan-400 pl-4 py-1"
          style={{ color: "#F8FAFC" }}
        >
          {highlightInsightText(
            dynamicInsight ||
              `Atmospheric conditions over ${state} are being analyzed. Click a high-risk column for detailed SHAP + LIME insights.`,
            district,
            state,
          )}
        </div>
        {/* Driver badges */}
        {activeGrid.primary_driver &&
          activeGrid.primary_driver !== "Stable" && (
            <div className="flex items-center space-x-2.5 mt-4 pt-1 flex-wrap gap-y-2">
              <span className="text-[13.5px] font-bold bg-red-500/15 border border-red-500/30 text-red-300 px-3 py-1.5 rounded-lg font-sans shadow-sm">
                Primary: {activeGrid.primary_driver}
              </span>
              {activeGrid.secondary_driver &&
                activeGrid.secondary_driver !== "Stable" && (
                  <span className="text-[13.5px] font-bold bg-amber-500/15 border border-amber-500/30 text-amber-300 px-3 py-1.5 rounded-lg font-sans shadow-sm">
                    Secondary: {activeGrid.secondary_driver}
                  </span>
                )}
            </div>
          )}
      </div>

      {/* ═══════ Historical Synoptic Pattern Match ═══════ */}
      <div className="border border-slate-700/50 bg-slate-800/40 rounded-xl p-5">
        <h3 className="text-sm font-bold text-slate-300 uppercase mb-3 flex items-center tracking-wider">
          <History className="w-4 h-4 mr-2 text-slate-400" />
          Synoptic Pattern Match
        </h3>
        {prob < 0.4 ||
        Number(similarityScore) === 0 ||
        eventMatch.includes("Nominal Atmospheric Baseline") ? (
          <div className="flex items-center justify-between mt-2 bg-slate-800/60 border border-slate-700/50 rounded-lg px-3.5 py-3 shadow-inner">
            <div className="flex items-center space-x-2.5">
              <div className="w-2.5 h-2.5 rounded-full bg-emerald-400" />
              <span className="text-[14px] font-medium text-emerald-400">
                Nominal Atmospheric Baseline
              </span>
            </div>
            <span className="text-[13px] font-medium text-slate-400">
              No Synoptic Threat
            </span>
          </div>
        ) : (
          <div className="flex items-end justify-between mt-2">
            <div className="text-4xl font-mono font-bold text-amber-400">
              {similarityScore}%
            </div>
            <div className="text-sm font-sans text-slate-300 max-w-[65%] text-right leading-snug">
              similarity to{" "}
              <span className="text-white font-bold">{eventMatch}</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
