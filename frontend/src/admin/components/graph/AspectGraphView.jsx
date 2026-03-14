import { useEffect, useMemo, useRef, useState } from "react";
import { DataSet } from "vis-data";
import { Network } from "vis-network";

import EdgeDetailsPanel from "./EdgeDetailsPanel";
import GraphLegend from "./GraphLegend";
import NodeDetailsPanel from "./NodeDetailsPanel";
import ReviewEvidencePanel from "./ReviewEvidencePanel";

const sentimentColors = {
  positive: "#22c55e",
  neutral: "#94a3b8",
  negative: "#ef4444",
  mixed: "#f59e0b",
};

function clamp(value, min, max) {
  return Math.max(min, Math.min(value, max));
}

function edgeColor(edge) {
  return sentimentColors[edge.polarity_hint] || "#38bdf8";
}

function normalizeNode(rawNode, index, scope) {
  if (!rawNode || typeof rawNode !== "object") return null;

  const fallbackId = `node-${index}`;
  const id = String(rawNode.id ?? rawNode.label ?? fallbackId).trim() || fallbackId;
  const label = String(rawNode.label ?? rawNode.id ?? fallbackId).trim() || fallbackId;
  const sentimentKey = scope === "single_review" ? rawNode.sentiment : rawNode.dominant_sentiment;
  const confidence = Number(rawNode.confidence ?? 0);
  const frequency = Number(rawNode.frequency ?? 1);
  const evidenceStart = Number(rawNode.evidence_start);
  const evidenceEnd = Number(rawNode.evidence_end);

  return {
    ...rawNode,
    id,
    label,
    sentiment: typeof rawNode.sentiment === "string" ? rawNode.sentiment.toLowerCase() : rawNode.sentiment,
    dominant_sentiment:
      typeof rawNode.dominant_sentiment === "string" ? rawNode.dominant_sentiment.toLowerCase() : rawNode.dominant_sentiment,
    polarity_hint: typeof rawNode.polarity_hint === "string" ? rawNode.polarity_hint.toLowerCase() : rawNode.polarity_hint,
    confidence: Number.isFinite(confidence) ? confidence : 0,
    frequency: Number.isFinite(frequency) ? frequency : 1,
    evidence: typeof rawNode.evidence === "string" ? rawNode.evidence : null,
    evidence_start: Number.isFinite(evidenceStart) ? evidenceStart : null,
    evidence_end: Number.isFinite(evidenceEnd) ? evidenceEnd : null,
    _sentiment_key:
      typeof sentimentKey === "string" && sentimentKey.trim() ? sentimentKey.toLowerCase() : "neutral",
  };
}

function normalizeGraph(graph, scope) {
  const rawNodes = Array.isArray(graph?.nodes) ? graph.nodes : [];
  const normalizedNodes = rawNodes
    .map((node, index) => normalizeNode(node, index, scope))
    .filter(Boolean);

  const validNodeIds = new Set(normalizedNodes.map((node) => node.id));
  const rawEdges = Array.isArray(graph?.edges) ? graph.edges : [];
  const normalizedEdges = rawEdges
    .map((edge, index) => {
      if (!edge || typeof edge !== "object") return null;
      const source = String(edge.source ?? "").trim();
      const target = String(edge.target ?? "").trim();
      if (!source || !target || !validNodeIds.has(source) || !validNodeIds.has(target)) {
        return null;
      }

      const weight = Number(edge.weight ?? 1);
      return {
        ...edge,
        id: `${source}-${target}-${index}`,
        source,
        target,
        weight: Number.isFinite(weight) ? weight : 1,
        directional: Boolean(edge.directional),
        polarity_hint: typeof edge.polarity_hint === "string" ? edge.polarity_hint.toLowerCase() : null,
      };
    })
    .filter(Boolean);

  return {
    ...graph,
    nodes: normalizedNodes,
    edges: normalizedEdges,
  };
}

function buildVisData(normalizedGraph, scope, isDark) {
  const nodes = normalizedGraph?.nodes || [];
  const edges = normalizedGraph?.edges || [];
  const maxFrequency = Math.max(...nodes.map((node) => Number(node.frequency || 1)), 1);
  const maxWeight = Math.max(...edges.map((edge) => Number(edge.weight || 1)), 1);
  const labelById = new Map(nodes.map((n) => [n.id, n.label]));

  const visNodes = nodes.map((node) => {
    const size =
      scope === "single_review"
        ? clamp(16 + Number(node.confidence || 0) * 10, 16, 28)
        : clamp(14 + (Number(node.frequency || 1) / maxFrequency) * 16, 14, 30);

    return {
      id: node.id,
      label: node.label,
      size,
      color: {
        background: sentimentColors[node._sentiment_key] || sentimentColors.neutral,
        border: isDark ? "#0f172a" : "#ffffff",
        highlight: {
          background: sentimentColors[node._sentiment_key] || sentimentColors.neutral,
          border: "#f59e0b",
        },
      },
      font: {
        color: isDark ? "#e5eefc" : "#10223d",
        size: 14,
        face: "Segoe UI",
        strokeWidth: 0,
      },
      payload: node,
    };
  });

  const visEdges = edges.map((edge) => ({
    id: edge.id,
    from: edge.source,
    to: edge.target,
    width: clamp(1 + (Number(edge.weight || 1) / maxWeight) * 4, 1, 5),
    color: {
      color: edgeColor(edge),
      opacity: 0.9,
      highlight: "#f59e0b",
    },
    arrows: scope === "single_review" ? "to" : "",
    smooth: { enabled: true, type: "dynamic" },
    payload: {
      ...edge,
      sourceLabel: labelById.get(edge.source) || edge.source,
      targetLabel: labelById.get(edge.target) || edge.target,
    },
  }));

  return {
    nodes: visNodes,
    edges: visEdges,
  };
}

export default function AspectGraphView({
  graph,
  scope = "batch",
  isDark = false,
  reviewText = "",
  emptyMessage = "No graph data available yet.",
}) {
  const [selection, setSelection] = useState({ type: "node", data: null });
  const containerRef = useRef(null);
  const networkRef = useRef(null);

  const normalizedGraph = useMemo(() => normalizeGraph(graph, scope), [graph, scope]);
  const hasGraph = Boolean((normalizedGraph?.nodes || []).length);
  const visData = useMemo(() => buildVisData(normalizedGraph, scope, isDark), [normalizedGraph, scope, isDark]);

  useEffect(() => {
    setSelection({ type: "node", data: null });
  }, [normalizedGraph, scope]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container || !hasGraph) {
      if (networkRef.current) {
        networkRef.current.destroy();
        networkRef.current = null;
      }
      return;
    }

    if (networkRef.current) {
      networkRef.current.destroy();
      networkRef.current = null;
    }

    const nodes = new DataSet(visData.nodes);
    const edges = new DataSet(visData.edges);

    const options = {
      autoResize: true,
      interaction: {
        hover: true,
        multiselect: false,
        dragNodes: true,
        dragView: true,
        zoomView: true,
      },
      nodes: {
        shape: "dot",
      },
      edges: {
        selectionWidth: 2,
      },
      physics: {
        enabled: true,
        stabilization: { enabled: true, iterations: scope === "single_review" ? 180 : 240, fit: true },
        barnesHut: {
          gravitationalConstant: scope === "single_review" ? -2200 : -2800,
          springLength: scope === "single_review" ? 130 : 150,
          springConstant: 0.04,
          damping: 0.25,
        },
      },
    };

    const network = new Network(container, { nodes, edges }, options);
    networkRef.current = network;

    network.on("click", (params) => {
      const nodeId = params.nodes?.[0];
      const edgeId = params.edges?.[0];

      if (nodeId) {
        const node = nodes.get(nodeId);
        setSelection({ type: "node", data: node?.payload || null });
        return;
      }

      if (edgeId) {
        const edge = edges.get(edgeId);
        setSelection({ type: "edge", data: edge?.payload || null });
        return;
      }

      setSelection({ type: "node", data: null });
    });

    network.on("dragEnd", (params) => {
      if (!params.nodes?.length) return;
      const positions = network.getPositions(params.nodes);
      params.nodes.forEach((nodeId) => {
        const pos = positions[nodeId];
        if (!pos) return;
        nodes.update({ id: nodeId, x: pos.x, y: pos.y, fixed: { x: true, y: true } });
      });
    });

    network.once("stabilizationIterationsDone", () => {
      network.setOptions({ physics: false });
      network.fit({ animation: { duration: 350, easingFunction: "easeInOutQuad" } });
    });

    return () => {
      network.destroy();
      if (networkRef.current === network) networkRef.current = null;
    };
  }, [visData, hasGraph, scope]);

  return (
    <div className="space-y-4">
      <GraphLegend scope={scope} isDark={isDark} />

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.55fr)_360px]">
        <div className={`overflow-hidden rounded-[28px] border ${isDark ? "border-slate-800 bg-[#071120]" : "border-slate-200 bg-white"}`}>
          <div className={`flex items-center justify-between border-b px-5 py-4 ${isDark ? "border-slate-800" : "border-slate-200"}`}>
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.18em] text-cyan-500">
                {scope === "single_review" ? "Single Review Explanation Graph" : "Corpus Aspect Graph"}
              </p>
              <p className={`mt-1 text-sm ${isDark ? "text-slate-400" : "text-slate-500"}`}>
                {scope === "single_review"
                  ? "Directed edges follow evidence order inside one review."
                  : "Undirected edges connect aspects that co-occur across reviews."}
              </p>
            </div>
            <div className={`text-xs uppercase tracking-[0.18em] ${isDark ? "text-slate-500" : "text-slate-400"}`}>
              {(normalizedGraph?.nodes || []).length} nodes / {(normalizedGraph?.edges || []).length} edges
            </div>
          </div>

          {hasGraph ? (
            <div ref={containerRef} className="h-[540px] w-full" />
          ) : (
            <div className={`grid h-[540px] place-items-center px-6 text-center text-sm ${isDark ? "text-slate-400" : "text-slate-500"}`}>
              {emptyMessage}
            </div>
          )}
        </div>

        <div className="space-y-4">
          <NodeDetailsPanel node={selection.type === "node" ? selection.data : null} scope={scope} isDark={isDark} />
          <EdgeDetailsPanel edge={selection.type === "edge" ? selection.data : null} scope={scope} isDark={isDark} />
        </div>
      </div>

      {scope === "single_review" ? <ReviewEvidencePanel text={reviewText} nodes={normalizedGraph?.nodes || []} isDark={isDark} /> : null}
    </div>
  );
}
