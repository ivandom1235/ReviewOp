import DataGridTable from "../components/DataGridTable";

function Stat({ label, value }) {
  return (
    <div className="glass-card rounded-2xl p-5 transition-all hover:translate-y-[-2px] hover:shadow-lg">
      <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-muted-main opacity-60">
        {label}
      </p>
      <p className="mt-2 text-3xl font-extrabold tracking-tight">
        {value}
      </p>
    </div>
  );
}

function Section({ title, children, className = "" }) {
  return (
    <div className={`glass-card rounded-2xl p-6 ${className}`}>
      <h3 className="mb-4 text-lg font-bold tracking-tight">{title}</h3>
      {children}
    </div>
  );
}

export default function Dashboard({ 
  kpis, 
  alerts, 
  leaderboardRows, 
  impactRows = [], 
  segmentRows = [], 
  weeklySummary, 
  onSeeMoreAlerts 
}) {
  return (
    <section className="space-y-6 page-fade-in">
      {/* KPI Grid */}
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-5">
        <Stat label="Total Reviews" value={kpis?.total_reviews ?? "-"} />
        <Stat label="Total Aspects" value={kpis?.total_aspects ?? "-"} />
        <Stat label="Most Negative" value={kpis?.most_negative_aspect ?? "-"} />
        <Stat label="% Negative" value={kpis ? `${kpis.negative_sentiment_pct}%` : "-"} />
        <Stat label="Emerging Issues" value={kpis?.emerging_issues_count ?? 0} />
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        {/* Weekly Summary */}
        <Section title="Weekly Insights" className="lg:col-span-1">
          {weeklySummary ? (
            <div className="space-y-4">
              <div className="rounded-xl border border-border-subtle bg-app/50 p-4">
                <p className="text-[11px] font-bold uppercase tracking-wider text-brand-primary">
                  Period: {weeklySummary.period_label}
                </p>
                <div className="mt-3 space-y-2 text-sm leading-relaxed">
                  <p><span className="font-semibold text-muted-main">Top Drivers:</span> {(weeklySummary.top_drivers || []).join(", ") || "-"}</p>
                  <p><span className="font-semibold text-muted-main">Rising Trend:</span> {weeklySummary.biggest_increase_aspect || "-"} (+{Number(weeklySummary.biggest_increase_pct || 0).toFixed(1)}%)</p>
                </div>
              </div>
              <div className="flex items-center gap-3 rounded-xl border border-brand-success/20 bg-brand-success/5 p-3 text-sm">
                <div className="h-2 w-2 animate-pulse rounded-full bg-brand-success" />
                <p className="font-medium">{weeklySummary.emerging_count || 0} New Emerging Issues Detected</p>
              </div>
            </div>
          ) : (
            <p className="py-8 text-center text-sm text-muted-main opacity-50">No data for this period.</p>
          )}
        </Section>

        {/* Alerts Panel */}
        <Section title="Critical Alerts" className="lg:col-span-2">
          <div className="space-y-3">
            {(alerts || []).length ? (
              <>
                <div className="grid gap-2">
                  {alerts.slice(0, 5).map((a, idx) => (
                    <div 
                      key={`${a.aspect}-${idx}`} 
                      className="flex items-center justify-between rounded-xl border border-border-subtle bg-app/30 p-3 transition-colors hover:bg-app/50"
                    >
                      <div className="flex items-center gap-3">
                        <span className={`rounded-full px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider ${
                          a.severity === 'high' ? 'bg-brand-danger/10 text-brand-danger' : 'bg-brand-primary/10 text-brand-primary'
                        }`}>
                          {a.severity}
                        </span>
                        <span className="text-sm font-medium">{a.message}</span>
                      </div>
                      <span className="text-xs font-bold text-muted-main opacity-40">{a.value}%</span>
                    </div>
                  ))}
                </div>
                {alerts.length > 5 && (
                  <button
                    onClick={onSeeMoreAlerts}
                    className="mt-2 w-full rounded-xl py-2 text-xs font-bold uppercase tracking-widest text-brand-primary transition-all hover:bg-brand-primary/5"
                  >
                    View All Anomalies →
                  </button>
                )}
              </>
            ) : (
              <p className="py-8 text-center text-sm text-muted-main opacity-50">System clear. No active alerts.</p>
            )}
          </div>
        </Section>
      </div>

      {/* Leaderboard Section */}
      <Section title="Aspect Performance Leaderboard">
        <DataGridTable
          height={400}
          rows={leaderboardRows}
          columns={[
            { field: "aspect", headerName: "Aspect", flex: 1.1, minWidth: 160 },
            { field: "frequency", headerName: "Freq", type: "number", flex: 0.5, minWidth: 80 },
            { field: "sample_size", headerName: "N", type: "number", flex: 0.4, minWidth: 70 },
            { field: "mentions_per_100_reviews", headerName: "Mentions/100", type: "number", flex: 0.7, minWidth: 110 },
            { field: "positive_pct", headerName: "Positive %", type: "number", flex: 0.6, minWidth: 100 },
            { field: "neutral_pct", headerName: "Neutral %", type: "number", flex: 0.6, minWidth: 100 },
            { field: "negative_pct", headerName: "Negative %", type: "number", flex: 0.6, minWidth: 100 },
            { field: "negative_ci_low", headerName: "Lower Bound", type: "number", flex: 0.6, minWidth: 100 },
            { field: "negative_ci_high", headerName: "Upper Bound", type: "number", flex: 0.6, minWidth: 100 },
            { field: "net_sentiment", headerName: "Net Score", type: "number", flex: 0.6, minWidth: 100 },
            { field: "change_7d_vs_prev_7d", headerName: "Trend (7d)", type: "number", flex: 0.8, minWidth: 130 },
            { field: "implicit_pct", headerName: "Implicit %", type: "number", flex: 0.6, minWidth: 100 },
          ]}
        />
      </Section>

      <div className="grid gap-6 xl:grid-cols-2">
        {/* Impact Matrix */}
        <Section title="Priority Fix Matrix">
          <DataGridTable
            height={320}
            rows={impactRows.map((r, i) => ({ id: `${r.aspect}-${i}`, ...r }))}
            columns={[
              { field: "aspect", headerName: "Aspect", flex: 1.2, minWidth: 140 },
              { field: "volume", headerName: "Volume", type: "number", flex: 0.5, minWidth: 80 },
              { field: "negative_rate", headerName: "Neg Rate", type: "number", flex: 0.6, minWidth: 90 },
              { field: "priority_score", headerName: "Priority", type: "number", flex: 0.6, minWidth: 90 },
              { 
                field: "action_tier", 
                headerName: "Tier", 
                flex: 0.6, 
                minWidth: 80,
                renderCell: (params) => (
                  <span className={`font-bold uppercase tracking-tighter ${
                    params.value === 'high' ? 'text-brand-danger' : 'text-brand-primary'
                  }`}>
                    {params.value}
                  </span>
                )
              },
            ]}
          />
        </Section>

        {/* Segment Drilldown */}
        <Section title="Segment Analysis">
          <DataGridTable
            height={320}
            rows={segmentRows.map((r, i) => ({ id: `${r.segment_type}-${r.segment_value}-${i}`, ...r }))}
            columns={[
              { field: "segment_type", headerName: "Type", flex: 0.5, minWidth: 80 },
              { field: "segment_value", headerName: "Segment", flex: 1, minWidth: 120 },
              { field: "mention_count", headerName: "Volume", type: "number", flex: 0.5, minWidth: 80 },
              { field: "negative_pct", headerName: "Neg %", type: "number", flex: 0.6, minWidth: 90 },
              { field: "top_negative_aspect", headerName: "Primary Issue", flex: 1, minWidth: 140 },
            ]}
          />
        </Section>
      </div>
    </section>
  );
}
