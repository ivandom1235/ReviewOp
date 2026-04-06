import { useState, useMemo } from "react";
import DataGridTable from "../components/DataGridTable";

export default function AlertsPage({ alerts = [], isDark, onAlertClick, onAlertClear }) {
  const [searchTerm, setSearchTerm] = useState("");
  const [severityFilter, setSeverityFilter] = useState("All");

  const filteredAlerts = useMemo(() => {
    return alerts.filter((a) => {
      const message = String(a?.message || "").toLowerCase();
      const aspect = String(a?.aspect || "").toLowerCase();
      const matchesSearch = 
        message.includes(searchTerm.toLowerCase()) || 
        aspect.includes(searchTerm.toLowerCase());
      const matchesSeverity = severityFilter === "All" || String(a?.severity || "").toLowerCase() === severityFilter.toLowerCase();
      return matchesSearch && matchesSeverity;
    });
  }, [alerts, searchTerm, severityFilter]);

  const rows = filteredAlerts.map((a, idx) => ({
    id: a.id ?? `${a.aspect}-${idx}`,
    ...a
  }));

  const columns = [
    { field: "severity", headerName: "Severity", width: 120, 
      renderCell: (params) => (
        <span className={`status-pill ${
          (params.value || "").toLowerCase() === "high" ? "status-pill-danger" :
          (params.value || "").toLowerCase() === "medium" ? "status-pill-warning" :
          "status-pill-info"
        }`}>
          {params.value}
        </span>
      )
    },
    { field: "aspect", headerName: "Aspect", width: 150 },
    { field: "message", headerName: "Message", flex: 1 },
    { 
      field: "action", 
      headerName: "Action",
      width: 180,
      renderCell: (params) => (
        <div className="flex items-center gap-2">
          <button 
            onClick={() => onAlertClick(params.row)}
            className="text-indigo-500 hover:text-indigo-400 font-semibold text-sm"
          >
            View
          </button>
          <button
            onClick={() => onAlertClear?.(params.row)}
            className="text-rose-500 hover:text-rose-400 font-semibold text-sm"
          >
            Clear
          </button>
        </div>
      )
    }
  ];

  return (
    <section className="space-y-4">
      <div className={`app-card rounded-2xl border p-6 ${isDark ? "border-slate-800 bg-[#0b1220]" : "border-slate-200 bg-white"}`}>
        <div className="flex flex-wrap items-center justify-between gap-4 mb-6">
          <h2 className="text-2xl font-bold">All System Alerts</h2>
          <div className="flex flex-wrap gap-3">
            <input 
              type="text" 
              placeholder="Search alerts..." 
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="app-input w-64"
            />
            <select 
              value={severityFilter}
              onChange={(e) => setSeverityFilter(e.target.value)}
              className="app-input w-44"
            >
              <option value="All">All Severities</option>
              <option value="high">high</option>
              <option value="medium">medium</option>
              <option value="low">low</option>
            </select>
          </div>
        </div>

        <DataGridTable 
          isDark={isDark}
          rows={rows}
          columns={columns}
          height={600}
        />
      </div>
    </section>
  );
}
