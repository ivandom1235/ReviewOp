import json
from collections import defaultdict
from typing import List, Dict

def find_disagreements(seq2seq_preds: List[Dict], protonet_preds: List[Dict]) -> List[Dict]:
    """
    Finds disagreements between Format A (Seq2Seq) and Format B (ProtoNet) proxy models.
    Assumes prediction format:
    [{"review_id": "rev_x", "aspect": "battery", "sentiment": "negative", "confidence": 0.8}, ...]
    """
    s2s = defaultdict(list)
    for p in seq2seq_preds:
        s2s[p["review_id"]].append(p)
        
    pn = defaultdict(list)
    for p in protonet_preds:
        pn[p["review_id"]].append(p)
        
    disagreements = []
    
    all_ids = set(s2s.keys()).union(set(pn.keys()))
    for rid in all_ids:
        # Sort to compare
        s2s_lbls = sorted([(x["aspect"], x.get("sentiment")) for x in s2s[rid]])
        pn_lbls = sorted([(x["aspect"], x.get("sentiment")) for x in pn[rid]])
        
        if s2s_lbls != pn_lbls:
            # High ambiguity if they are confident but disagree
            conf_s2s = sum(x.get("confidence", 0) for x in s2s[rid]) / max(1, len(s2s[rid]))
            conf_pn = sum(x.get("confidence", 0) for x in pn[rid]) / max(1, len(pn[rid]))
            
            disagreements.append({
                "review_id": rid,
                # Try getting the text if available, fallback to empty string
                "text": s2s[rid][0].get("text", pn[rid][0].get("text", "") if pn[rid] else "") if s2s[rid] else "",
                "seq2seq_prediction": s2s_lbls,
                "protonet_prediction": pn_lbls,
                "avg_confidence": (conf_s2s + conf_pn) / 2
            })
            
    # Sort by confidence (high confidence disagreements are more useful to review)
    return sorted(disagreements, key=lambda x: x["avg_confidence"], reverse=True)

def dump_for_human_review(disagreements: List[Dict], output_path: str = "active_learning_review.jsonl"):
    with open(output_path, "w", encoding="utf-8") as f:
        for d in disagreements:
            f.write(json.dumps(d) + "\n")
            
def generate_html_review_interface(disagreements: List[Dict], output_path: str = "review_interface.html"):
    """Creates a lightweight static HTML to review the disagreements."""
    html = ["<html><head><style>body{font-family:sans-serif; background:#f4f4f5;} .card{border:1px solid #ddd; background:#fff; margin-bottom:15px; padding:20px; border-radius:8px; box-shadow:0 2px 4px rgba(0,0,0,0.05);} .diff{color:crimson;} .seq{color:#007bff; font-weight:bold;} .proto{color:#28a745; font-weight:bold;} input[type=text] {padding:8px; width:400px;}</style></head><body>"]
    html.append("<div style='max-width:800px; margin:auto;'>")
    html.append("<h1>Active Learning Dashboard</h1>")
    html.append("<p>Review uncertainty samples where the proxy Seq2Seq model and ProtoNet explicitly disagree.</p>")
    
    for i, d in enumerate(disagreements):
        html.append(f"<div class='card'>")
        html.append(f"<h3>Review <code>{d['review_id']}</code> (Avg Confidence: {d['avg_confidence']:.2f})</h3>")
        html.append(f"<p style='font-size:1.1em; line-height:1.4;'>\"{d['text']}\"</p>")
        html.append(f"<p><span class='seq'>Seq2Seq Prediction:</span> {d['seq2seq_prediction']}</p>")
        html.append(f"<p><span class='proto'>ProtoNet Prediction:</span> {d['protonet_prediction']}</p>")
        html.append(f"<hr style='border:none; border-top:1px dashed #ccc;' />")
        html.append(f"<label><strong>Human Output (Aspect | Sentiment):</strong> <br><br><input type='text' placeholder='e.g., battery_life | negative' /></label>")
        html.append(f"</div>")
        
    html.append("</div></body></html>")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
