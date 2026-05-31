# app.py — Plant Health Detector (Production Website)
import torch
import clip
import torch.nn as nn
from PIL import Image
import json, os, io, base64
import numpy as np
from flask import Flask, request, jsonify, render_template_string

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — tweak these numbers freely
# ══════════════════════════════════════════════════════════════════════════════
MODEL_PATH        = "clip_plant_best.pt"
LABELS_PATH       = "clip_labels.json"
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"

# If fine-tuned model confidence is below this → fallback to zero-shot CLIP
FALLBACK_THRESHOLD = 70.0   # ← change this to whatever you want (0-100)

# Max image size before resizing (keeps inference fast)
MAX_IMAGE_SIZE    = 1024

# ══════════════════════════════════════════════════════════════════════════════
# DISEASE DATABASE
# ══════════════════════════════════════════════════════════════════════════════
DISEASE_DB = {
    "Tomato___Late_blight":                       {"cause":"Phytophthora infestans","severity":"HIGH","treatment":"Apply copper-based fungicide immediately. Remove and destroy infected leaves. Avoid overhead watering."},
    "Tomato___Early_blight":                      {"cause":"Alternaria solani","severity":"MEDIUM","treatment":"Apply chlorothalonil or mancozeb fungicide. Remove lower infected leaves. Stake plants for airflow."},
    "Tomato___Bacterial_spot":                    {"cause":"Xanthomonas spp.","severity":"MEDIUM","treatment":"Spray copper bactericide. Avoid working with wet plants. Use disease-free seeds."},
    "Tomato___Leaf_Mold":                         {"cause":"Passalora fulva","severity":"MEDIUM","treatment":"Reduce humidity below 85%. Apply fungicide. Prune lower leaves."},
    "Tomato___Septoria_leaf_spot":                {"cause":"Septoria lycopersici","severity":"MEDIUM","treatment":"Remove infected leaves. Apply mancozeb every 7-10 days."},
    "Tomato___Spider_mites Two-spotted_spider_mite":{"cause":"Tetranychus urticae","severity":"MEDIUM","treatment":"Spray neem oil or insecticidal soap. Increase humidity."},
    "Tomato___Target_Spot":                       {"cause":"Corynespora cassiicola","severity":"MEDIUM","treatment":"Apply azoxystrobin fungicide. Remove infected leaves."},
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":     {"cause":"TYLCV (whitefly-transmitted)","severity":"HIGH","treatment":"No cure. Remove infected plants. Control whiteflies with insecticide."},
    "Tomato___Tomato_mosaic_virus":               {"cause":"ToMV virus","severity":"HIGH","treatment":"No cure. Remove and destroy plant. Disinfect all tools with bleach."},
    "Tomato___healthy":                           {"cause":"None","severity":"NONE","treatment":"Plant is healthy. Continue regular monitoring."},
    "Potato___Early_blight":                      {"cause":"Alternaria solani","severity":"MEDIUM","treatment":"Apply chlorothalonil fungicide. Rotate crops every 3 years. Improve soil drainage."},
    "Potato___Late_blight":                       {"cause":"Phytophthora infestans","severity":"HIGH","treatment":"Apply metalaxyl + mancozeb immediately. Destroy all infected plants."},
    "Potato___healthy":                           {"cause":"None","severity":"NONE","treatment":"Plant is healthy. Continue regular monitoring."},
    "Apple___Apple_scab":                         {"cause":"Venturia inaequalis","severity":"MEDIUM","treatment":"Apply myclobutanil or captan fungicide in spring. Rake and destroy fallen leaves."},
    "Apple___Black_rot":                          {"cause":"Botryosphaeria obtusa","severity":"HIGH","treatment":"Prune infected branches 15cm below infection. Apply captan fungicide."},
    "Apple___Cedar_apple_rust":                   {"cause":"Gymnosporangium juniperi-virginianae","severity":"MEDIUM","treatment":"Apply myclobutanil in spring. Remove nearby juniper trees if possible."},
    "Apple___healthy":                            {"cause":"None","severity":"NONE","treatment":"Plant is healthy. Continue regular monitoring."},
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot":{"cause":"Cercospora zeae-maydis","severity":"MEDIUM","treatment":"Apply foliar fungicide. Plant resistant hybrids."},
    "Corn_(maize)___Common_rust_":                {"cause":"Puccinia sorghi","severity":"LOW","treatment":"Apply triazole fungicide if severe. Plant resistant varieties."},
    "Corn_(maize)___Northern_Leaf_Blight":        {"cause":"Exserohilum turcicum","severity":"MEDIUM","treatment":"Apply propiconazole fungicide at tasseling. Crop rotation."},
    "Corn_(maize)___healthy":                     {"cause":"None","severity":"NONE","treatment":"Plant is healthy. Continue regular monitoring."},
    "Grape___Black_rot":                          {"cause":"Guignardia bidwellii","severity":"HIGH","treatment":"Apply mancozeb or myclobutanil. Remove mummified berries."},
    "Grape___Esca_(Black_Measles)":               {"cause":"Fungal complex (Phaeomoniella)","severity":"HIGH","treatment":"No effective cure. Prune infected wood. Apply wound protectant."},
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {"cause":"Isariopsis clavispora","severity":"MEDIUM","treatment":"Apply copper-based fungicide. Improve canopy airflow."},
    "Grape___healthy":                            {"cause":"None","severity":"NONE","treatment":"Plant is healthy. Continue regular monitoring."},
    "Peach___Bacterial_spot":                     {"cause":"Xanthomonas arboricola","severity":"MEDIUM","treatment":"Apply copper bactericide in spring. Avoid overhead irrigation."},
    "Peach___healthy":                            {"cause":"None","severity":"NONE","treatment":"Plant is healthy. Continue regular monitoring."},
    "Pepper,_bell___Bacterial_spot":              {"cause":"Xanthomonas campestris","severity":"MEDIUM","treatment":"Spray copper bactericide. Use disease-free seeds."},
    "Pepper,_bell___healthy":                     {"cause":"None","severity":"NONE","treatment":"Plant is healthy. Continue regular monitoring."},
    "Squash___Powdery_mildew":                    {"cause":"Podosphaera xanthii","severity":"LOW","treatment":"Spray potassium bicarbonate or sulfur. Improve air circulation."},
    "Strawberry___Leaf_scorch":                   {"cause":"Diplocarpon earlianum","severity":"MEDIUM","treatment":"Apply captan fungicide. Remove old leaves. Improve drainage."},
    "Strawberry___healthy":                       {"cause":"None","severity":"NONE","treatment":"Plant is healthy. Continue regular monitoring."},
    "Cherry_(including_sour)___Powdery_mildew":   {"cause":"Podosphaera clandestina","severity":"LOW","treatment":"Apply sulfur spray. Prune for air circulation."},
    "Cherry_(including_sour)___healthy":          {"cause":"None","severity":"NONE","treatment":"Plant is healthy. Continue regular monitoring."},
    "Orange___Haunglongbing_(Citrus_greening)":   {"cause":"Candidatus Liberibacter","severity":"CRITICAL","treatment":"No cure. Remove and destroy infected tree immediately. Control Asian citrus psyllid."},
    "Raspberry___healthy":                        {"cause":"None","severity":"NONE","treatment":"Plant is healthy. Continue regular monitoring."},
    "Soybean___healthy":                          {"cause":"None","severity":"NONE","treatment":"Plant is healthy. Continue regular monitoring."},
    "Blueberry___healthy":                        {"cause":"None","severity":"NONE","treatment":"Plant is healthy. Continue regular monitoring."},
}

# Zero-shot prompts used as fallback when confidence < threshold
ZEROSHOT_PROMPTS = [
    "a healthy green plant leaf with no spots, no damage, completely normal",
    "a plant leaf with early blight disease showing brown concentric ring spots",
    "a plant leaf with late blight disease showing dark water-soaked lesions",
    "a tomato plant leaf infected with bacterial spot disease",
    "a potato leaf showing signs of early blight with Alternaria fungal infection",
    "a potato plant leaf infected with late blight Phytophthora disease",
    "a tomato leaf showing yellow leaf curl virus symptoms",
    "a grape leaf with black rot fungal disease spots",
    "a leaf covered with white powdery mildew coating on the surface",
    "a corn leaf with gray leaf spot or cercospora disease lesions",
    "a corn plant leaf with orange rust pustules from common rust disease",
    "a corn leaf with northern leaf blight showing elongated gray lesions",
    "an apple leaf with olive-green scab lesions from apple scab disease",
    "an apple leaf showing black rot disease symptoms",
    "a leaf showing signs of spider mite infestation with stippling damage",
    "a tomato leaf showing mosaic virus symptoms with mottled coloring",
    "a strawberry leaf with scorch disease showing purple spots",
    "a pepper leaf with bacterial spot disease",
    "a peach leaf showing bacterial spot disease symptoms",
    "a citrus leaf showing Huanglongbing greening disease",
    "a plant leaf showing septoria leaf spot disease",
    "a plant leaf showing target spot disease with concentric rings",
    "a plant leaf showing leaf mold disease with olive-green patches",
    "a grape leaf showing Esca black measles disease",
    "a grape leaf with isariopsis leaf blight symptoms",
    "an entire plant showing multiple diseased leaves from above",
    "multiple plant leaves photographed together showing disease",
    "a close-up photo of a diseased leaf with unclear focus",
    "a plant stem and leaves showing disease symptoms",
    "an overhead view of a plant with unhealthy yellowing leaves",
]

# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODEL AT STARTUP
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nLoading CLIP on {DEVICE}...")
model, preprocess = clip.load("ViT-B/16", device=DEVICE)
model = model.float()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("Fine-tuned model loaded ✓")

data    = json.load(open(LABELS_PATH))
CLASSES = data["classes"]
PROMPTS = data["prompts"]

# Precompute fine-tuned text embeddings (used when confidence >= threshold)
with torch.no_grad():
    text_tokens_ft    = clip.tokenize(PROMPTS, truncate=True).to(DEVICE)
    TEXT_FEATURES_FT  = model.encode_text(text_tokens_ft)
    TEXT_FEATURES_FT  = nn.functional.normalize(TEXT_FEATURES_FT, dim=-1)

# Precompute zero-shot text embeddings (used when confidence < threshold)
with torch.no_grad():
    text_tokens_zs    = clip.tokenize(ZEROSHOT_PROMPTS, truncate=True).to(DEVICE)
    TEXT_FEATURES_ZS  = model.encode_text(text_tokens_zs)
    TEXT_FEATURES_ZS /= TEXT_FEATURES_ZS.norm(dim=-1, keepdim=True)

print(f"Text embeddings ready — {len(CLASSES)} fine-tuned classes + {len(ZEROSHOT_PROMPTS)} zero-shot prompts ✓")
print(f"Fallback threshold: {FALLBACK_THRESHOLD}%\n")


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def run_finetuned(image_tensor):
    """Run fine-tuned CLIP — returns (probs array, classes list)"""
    with torch.no_grad():
        img_feat = model.encode_image(image_tensor)
        img_feat = nn.functional.normalize(img_feat, dim=-1)
        logits   = 100.0 * (img_feat @ TEXT_FEATURES_FT.T)
        probs    = logits.softmax(dim=-1)[0].cpu().numpy()
    return probs, CLASSES


def run_zeroshot(image_tensor):
    """Run zero-shot CLIP — returns (probs array, prompts list)"""
    with torch.no_grad():
        img_feat  = model.encode_image(image_tensor)
        txt_feat  = TEXT_FEATURES_ZS.clone()
        img_feat /= img_feat.norm(dim=-1, keepdim=True)
        similarity = (100.0 * img_feat @ txt_feat.T).softmax(dim=-1)
        probs      = similarity[0].cpu().numpy()
    return probs, ZEROSHOT_PROMPTS


def predict(pil_image: Image.Image, threshold: float = FALLBACK_THRESHOLD) -> dict:
    """
    Full prediction pipeline:
    1. Run fine-tuned model
    2. If top confidence < threshold → fallback to zero-shot
    """
    img_tensor = preprocess(pil_image).unsqueeze(0).to(DEVICE)

    # ── Step 1: Fine-tuned model ──────────────────────────────────────────────
    ft_probs, ft_classes = run_finetuned(img_tensor)
    top5_idx     = ft_probs.argsort()[::-1][:5]
    top_conf     = float(ft_probs[top5_idx[0]]) * 100
    top_label    = ft_classes[top5_idx[0]]

    parts      = top_label.split("___")
    plant      = parts[0].replace("_", " ")
    condition  = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"
    is_healthy = "healthy" in top_label.lower()

    top5 = [
        {
            "label":      ft_classes[i].replace("___", " — ").replace("_", " "),
            "confidence": round(float(ft_probs[i]) * 100, 2)
        }
        for i in top5_idx
    ]

    db_info  = DISEASE_DB.get(top_label, {})
    used_fallback = False

    # ── Step 2: Fallback if confidence is low ─────────────────────────────────
    zeroshot_results = []
    if top_conf < threshold:
        used_fallback = True
        zs_probs, zs_prompts = run_zeroshot(img_tensor)
        top3_zs = zs_probs.argsort()[::-1][:3]
        zeroshot_results = [
            {
                "description": zs_prompts[i],
                "confidence":  round(float(zs_probs[i]) * 100, 2)
            }
            for i in top3_zs
        ]

    return {
        "plant":            plant,
        "condition":        condition,
        "is_healthy":       is_healthy,
        "confidence":       round(top_conf, 2),
        "used_fallback":    used_fallback,
        "threshold_used":   threshold,
        "top5":             top5,
        "zeroshot_results": zeroshot_results,
        "disease_info": {
            "cause":     db_info.get("cause",     "Consult agricultural specialist"),
            "severity":  db_info.get("severity",  "MEDIUM"),
            "treatment": db_info.get("treatment", "Consult local agricultural extension office"),
        }
    }


# ══════════════════════════════════════════════════════════════════════════════
# HTML TEMPLATE
# ══════════════════════════════════════════════════════════════════════════════
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Plant Health Detector</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:         #0b1612;
  --card:       #111f17;
  --card2:      #162319;
  --border:     #1e3528;
  --green-dark: #1b4332;
  --green-mid:  #2d6a4f;
  --green-lite: #52b788;
  --green-pale: #95d5b2;
  --text:       #d8f3dc;
  --muted:      #6b9e7a;
  --amber:      #f0c040;
  --orange:     #ff8800;
  --red:        #ff4444;
  --red-crit:   #cc0000;
  --blue:       #4a9eff;
}

body {
  font-family: 'Segoe UI', system-ui, sans-serif;
  background: var(--bg);
  color: var(--text);
  min-height: 100vh;
}

/* ── HEADER ── */
header {
  background: linear-gradient(135deg, #0d2818, #1b4332 60%, #2d6a4f);
  padding: 18px 28px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  border-bottom: 1px solid #2d6a4f44;
}
.header-left  { display: flex; align-items: center; gap: 14px; }
.header-icon  { font-size: 34px; filter: drop-shadow(0 0 8px #52b78866); }
.header-title { font-size: 20px; font-weight: 700; color: #d8f3dc; }
.header-sub   { font-size: 11px; color: var(--muted); margin-top: 2px; }

.threshold-control {
  display: flex;
  align-items: center;
  gap: 10px;
  background: #0d2818;
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 8px 14px;
  font-size: 12px;
  color: var(--green-pale);
}
.threshold-control label { white-space: nowrap; }
.threshold-control input[type=range] {
  width: 100px;
  accent-color: var(--green-lite);
}
.threshold-control span {
  min-width: 38px;
  font-weight: 700;
  color: var(--amber);
  text-align: right;
}

/* ── LAYOUT ── */
.layout {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 18px;
  max-width: 1140px;
  margin: 24px auto;
  padding: 0 18px;
}

/* ── CARDS ── */
.card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 14px;
  overflow: hidden;
}
.card-head {
  background: var(--green-dark);
  padding: 11px 18px;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 1.2px;
  text-transform: uppercase;
  color: var(--green-pale);
  display: flex;
  align-items: center;
  gap: 7px;
}
.card-body { padding: 20px; }

/* ── DROP ZONE ── */
.drop-zone {
  border: 2px dashed var(--green-mid);
  border-radius: 12px;
  padding: 42px 20px;
  text-align: center;
  cursor: pointer;
  transition: all 0.2s;
  background: var(--bg);
  position: relative;
}
.drop-zone:hover, .drop-zone.over {
  border-color: var(--green-lite);
  background: #0d2818;
}
.drop-zone .dz-icon { font-size: 48px; margin-bottom: 10px; display: block; }
.drop-zone .dz-main { color: var(--text); font-size: 14px; font-weight: 500; }
.drop-zone .dz-sub  { color: var(--muted); font-size: 11px; margin-top: 5px; }

#file-input { display: none; }

/* ── FILE ACTIONS ── */
.file-actions {
  display: flex;
  gap: 10px;
  margin-top: 14px;
}
.btn-secondary {
  flex: 1;
  padding: 10px;
  background: transparent;
  color: var(--green-pale);
  border: 1px solid var(--border);
  border-radius: 8px;
  font-size: 13px;
  cursor: pointer;
  transition: all 0.2s;
}
.btn-secondary:hover { border-color: var(--green-lite); color: var(--green-lite); }

/* ── PREVIEW ── */
#preview-wrap {
  display: none;
  margin-top: 14px;
  border-radius: 10px;
  overflow: hidden;
  border: 1px solid var(--border);
  background: #0b1612;
  text-align: center;
}
#preview-wrap img {
  max-width: 100%;
  max-height: 270px;
  object-fit: contain;
  display: block;
  margin: 0 auto;
}
.img-meta {
  padding: 7px 12px;
  font-size: 10px;
  color: var(--muted);
  border-top: 1px solid var(--border);
  display: flex;
  justify-content: space-between;
}

/* ── ANALYSE BUTTON ── */
#analyse-btn {
  display: none;
  width: 100%;
  margin-top: 14px;
  padding: 14px;
  background: linear-gradient(135deg, var(--green-dark), var(--green-mid));
  color: var(--text);
  border: none;
  border-radius: 10px;
  font-size: 15px;
  font-weight: 700;
  cursor: pointer;
  transition: all 0.2s;
  letter-spacing: 0.5px;
}
#analyse-btn:hover    { background: linear-gradient(135deg, var(--green-mid), var(--green-lite)); color: #0b1612; }
#analyse-btn:disabled { opacity: 0.5; cursor: default; }

/* ── EMPTY / SPINNER ── */
.empty-state {
  text-align: center;
  padding: 60px 20px;
  color: var(--muted);
}
.empty-state .e-icon { font-size: 52px; margin-bottom: 14px; opacity: 0.5; }
.empty-state p { font-size: 13px; line-height: 1.6; }

.spinner-wrap {
  display: none;
  text-align: center;
  padding: 60px;
}
.spinner {
  width: 40px; height: 40px;
  border: 3px solid var(--border);
  border-top-color: var(--green-lite);
  border-radius: 50%;
  animation: spin 0.7s linear infinite;
  margin: 0 auto 14px;
}
@keyframes spin { to { transform: rotate(360deg); } }
.spinner-wrap p { color: var(--muted); font-size: 13px; }

/* ── RESULT ── */
#result-panel { display: none; }

/* Status badges */
.status-row { margin-bottom: 16px; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.badge {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 5px 14px;
  border-radius: 20px;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.3px;
}
.badge-healthy  { background: #0d2818; color: var(--green-lite); border: 1px solid var(--green-mid); }
.badge-diseased { background: #2a1200; color: var(--orange); border: 1px solid #6b3010; }
.badge-fallback { background: #1a1a00; color: var(--amber); border: 1px solid #6b5500; font-size: 10px; }

/* Info rows */
.info-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 9px 0;
  border-bottom: 1px solid var(--border);
  font-size: 13px;
}
.info-row:last-child { border: none; }
.info-label { color: var(--muted); font-size: 11px; }
.info-val   { font-weight: 500; color: var(--text); text-align: right; max-width: 65%; }

/* Confidence bar */
.conf-section { margin: 14px 0; }
.conf-header  {
  display: flex; justify-content: space-between;
  font-size: 11px; color: var(--muted); margin-bottom: 6px;
}
.conf-header strong { color: var(--amber); font-size: 13px; }
.conf-track { height: 10px; background: var(--border); border-radius: 5px; overflow: hidden; }
.conf-fill  {
  height: 100%;
  border-radius: 5px;
  transition: width 0.7s cubic-bezier(.4,0,.2,1);
}

/* Disease info box */
.disease-box {
  border-radius: 10px;
  padding: 14px;
  margin: 14px 0;
  font-size: 13px;
  line-height: 1.7;
  border-left: 4px solid;
}
.disease-box .db-row { margin: 3px 0; }
.disease-box .db-row strong { color: var(--green-pale); }

.sev-pill {
  display: inline-block;
  padding: 2px 10px;
  border-radius: 10px;
  font-size: 10px;
  font-weight: 700;
  margin-left: 6px;
  vertical-align: middle;
}

/* Top 5 */
.top5-section { margin-top: 18px; }
.section-label {
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 1px;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 10px;
}
.top5-row {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 6px 0;
  font-size: 12px;
}
.top5-rank  { min-width: 16px; color: var(--muted); }
.top5-name  { flex: 1; color: var(--green-pale); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.bar-track  { width: 90px; height: 5px; background: var(--border); border-radius: 3px; overflow: hidden; flex-shrink: 0; }
.bar-fill   { height: 100%; background: var(--green-mid); border-radius: 3px; transition: width 0.6s; }
.top5-pct   { min-width: 40px; text-align: right; color: var(--muted); }

/* Fallback zero-shot results */
.fallback-section {
  margin-top: 18px;
  background: #1a1a05;
  border: 1px solid #5a5500;
  border-radius: 10px;
  padding: 14px;
}
.fallback-title {
  font-size: 11px;
  font-weight: 700;
  color: var(--amber);
  letter-spacing: 0.5px;
  margin-bottom: 10px;
  display: flex;
  align-items: center;
  gap: 6px;
}
.fallback-row {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 6px 0;
  font-size: 12px;
  color: #c0b040;
}
.fallback-desc { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.fallback-pct  { min-width: 40px; text-align: right; font-weight: 600; }

/* Healthy box */
.healthy-box {
  background: #081a10;
  border: 1px solid var(--green-mid);
  border-left: 4px solid var(--green-lite);
  border-radius: 10px;
  padding: 14px;
  margin: 14px 0;
  font-size: 13px;
  color: var(--green-lite);
  line-height: 1.6;
}

/* ── RESPONSIVE ── */
@media (max-width: 720px) {
  .layout { grid-template-columns: 1fr; }
  .threshold-control { display: none; }
}
</style>
</head>
<body>

<header>
  <div class="header-left">
    <div class="header-icon">🌿</div>
    <div>
      <div class="header-title">Plant Health Detector</div>
      <div class="header-sub">CLIP Fine-tuned 98.45% · 38 classes · RTX 3050 GPU</div>
    </div>
  </div>
  <div class="threshold-control">
    <label>Fallback threshold</label>
    <input type="range" id="threshold-slider" min="0" max="100" step="5" value="{{ threshold }}">
    <span id="threshold-val">{{ threshold }}%</span>
  </div>
</header>

<div class="layout">

  <!-- ── LEFT: UPLOAD ── -->
  <div class="card">
    <div class="card-head">📷 Upload Plant Image</div>
    <div class="card-body">

      <div class="drop-zone" id="drop-zone"
           onclick="document.getElementById('file-input').click()">
        <span class="dz-icon">🍃</span>
        <div class="dz-main">Click to select or drag & drop image here</div>
        <div class="dz-sub">JPG · JPEG · PNG · BMP · WEBP · TIFF — any format</div>
      </div>

      <input type="file" id="file-input"
             accept="image/*,.jpg,.jpeg,.png,.bmp,.webp,.tiff,.tif">

      <div class="file-actions">
        <button class="btn-secondary" onclick="document.getElementById('file-input').click()">
          📁 Browse Files
        </button>
        <button class="btn-secondary" id="clear-btn" onclick="clearImage()" style="display:none">
          🗑 Clear
        </button>
      </div>

      <div id="preview-wrap">
        <img id="preview" src="" alt="Preview">
        <div class="img-meta">
          <span id="img-name">—</span>
          <span id="img-size">—</span>
        </div>
      </div>

      <button id="analyse-btn" onclick="analyse()">🔍 Analyse Plant</button>

    </div>
  </div>

  <!-- ── RIGHT: RESULT ── -->
  <div class="card">
    <div class="card-head">📊 Analysis Result</div>
    <div class="card-body">

      <div class="empty-state" id="empty-state">
        <div class="e-icon">🔬</div>
        <p>Upload any plant or leaf image<br>
           to detect species and diseases.<br><br>
           <span style="font-size:11px;color:#4a7a5a">
             Works with single leaves, multiple leaves,<br>
             full plants, any angle or zoom level.
           </span>
        </p>
      </div>

      <div class="spinner-wrap" id="spinner-wrap">
        <div class="spinner"></div>
        <p id="spinner-msg">Running CLIP inference...</p>
      </div>

      <div id="result-panel"></div>

    </div>
  </div>

</div>

<script>
  // ── Threshold slider ──────────────────────────────────────────────────────
  const slider    = document.getElementById('threshold-slider');
  const threshVal = document.getElementById('threshold-val');
  slider.addEventListener('input', () => {
    threshVal.textContent = slider.value + '%';
  });

  // ── Drag and drop ─────────────────────────────────────────────────────────
  const dz = document.getElementById('drop-zone');
  dz.addEventListener('dragover',  e => { e.preventDefault(); dz.classList.add('over'); });
  dz.addEventListener('dragleave', () => dz.classList.remove('over'));
  dz.addEventListener('drop', e => {
    e.preventDefault();
    dz.classList.remove('over');
    const f = e.dataTransfer.files[0];
    if (f) loadFile(f);
  });

  document.getElementById('file-input').addEventListener('change', function() {
    if (this.files[0]) loadFile(this.files[0]);
  });

  // ── Paste from clipboard ──────────────────────────────────────────────────
  document.addEventListener('paste', e => {
    const items = e.clipboardData.items;
    for (let item of items) {
      if (item.type.startsWith('image/')) {
        loadFile(item.getAsFile());
        break;
      }
    }
  });

  let selectedFile = null;

  function loadFile(file) {
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = ev => {
      document.getElementById('preview').src = ev.target.result;
      document.getElementById('preview-wrap').style.display = 'block';
      document.getElementById('img-name').textContent = file.name || 'pasted image';
      document.getElementById('img-size').textContent = (file.size/1024).toFixed(1) + ' KB';
    };
    reader.readAsDataURL(file);

    document.getElementById('analyse-btn').style.display = 'block';
    document.getElementById('clear-btn').style.display   = 'block';
    resetResult();
  }

  function clearImage() {
    selectedFile = null;
    document.getElementById('file-input').value = '';
    document.getElementById('preview-wrap').style.display = 'none';
    document.getElementById('analyse-btn').style.display  = 'none';
    document.getElementById('clear-btn').style.display    = 'none';
    resetResult();
  }

  function resetResult() {
    document.getElementById('empty-state').style.display  = 'block';
    document.getElementById('spinner-wrap').style.display = 'none';
    document.getElementById('result-panel').style.display = 'none';
    document.getElementById('result-panel').innerHTML     = '';
  }

  // ── Analyse ───────────────────────────────────────────────────────────────
  async function analyse() {
    if (!selectedFile) return;

    const btn       = document.getElementById('analyse-btn');
    const threshold = parseFloat(slider.value);

    btn.disabled    = true;
    btn.textContent = '⏳ Analysing...';
    document.getElementById('empty-state').style.display  = 'none';
    document.getElementById('result-panel').style.display = 'none';
    document.getElementById('spinner-wrap').style.display = 'block';
    document.getElementById('spinner-msg').textContent    = 'Running fine-tuned CLIP model...';

    const form = new FormData();
    form.append('image', selectedFile);
    form.append('threshold', threshold);

    try {
      const resp = await fetch('/predict', { method: 'POST', body: form });
      const data = await resp.json();
      if (data.error) throw new Error(data.error);

      if (data.used_fallback) {
        document.getElementById('spinner-msg').textContent =
          `Confidence ${data.confidence.toFixed(1)}% < ${threshold}% threshold → running zero-shot CLIP...`;
        await new Promise(r => setTimeout(r, 600));
      }

      renderResult(data);
    } catch(e) {
      document.getElementById('result-panel').innerHTML =
        `<div style="color:#ff6666;padding:20px;font-size:14px">❌ Error: ${e.message}</div>`;
      document.getElementById('result-panel').style.display = 'block';
    }

    document.getElementById('spinner-wrap').style.display = 'none';
    btn.disabled    = false;
    btn.textContent = '🔍 Analyse Plant';
  }

  // ── Render result ─────────────────────────────────────────────────────────
  const SEV = {
    CRITICAL: { bg:'#3a0000', text:'#ff4444', border:'#cc0000', pill:'#ff444433' },
    HIGH:     { bg:'#2a1200', text:'#ff8800', border:'#cc6600', pill:'#ff880033' },
    MEDIUM:   { bg:'#2a2000', text:'#f0c040', border:'#aa8800', pill:'#f0c04033' },
    LOW:      { bg:'#0d2018', text:'#52b788', border:'#2d6a4f', pill:'#52b78833' },
    NONE:     { bg:'#0d2018', text:'#52b788', border:'#2d6a4f', pill:'#52b78833' },
  };

  function confColor(c) {
    if (c >= 85) return 'linear-gradient(90deg,#2d6a4f,#52b788)';
    if (c >= 70) return 'linear-gradient(90deg,#6b8800,#f0c040)';
    return 'linear-gradient(90deg,#8b2000,#ff8800)';
  }

  function renderResult(d) {
    const panel = document.getElementById('result-panel');
    const conf  = d.confidence;
    const sev   = d.disease_info ? d.disease_info.severity : 'NONE';
    const sc    = SEV[sev] || SEV['MEDIUM'];
    let   html  = '';

    // Status badges
    html += `<div class="status-row">`;
    if (d.is_healthy) {
      html += `<span class="badge badge-healthy">✅ HEALTHY PLANT</span>`;
    } else {
      html += `<span class="badge badge-diseased">⚠️ DISEASE DETECTED</span>`;
    }
    if (d.used_fallback) {
      html += `<span class="badge badge-fallback">
        ⚡ ZERO-SHOT FALLBACK (confidence was ${conf.toFixed(1)}%)
      </span>`;
    }
    html += `</div>`;

    // Info rows
    html += `
      <div class="info-row">
        <span class="info-label">Plant Species</span>
        <span class="info-val">${d.plant}</span>
      </div>
      <div class="info-row">
        <span class="info-label">Condition</span>
        <span class="info-val">${d.condition}</span>
      </div>
      <div class="info-row">
        <span class="info-label">Analysis Mode</span>
        <span class="info-val" style="color:${d.used_fallback ? 'var(--amber)' : 'var(--green-lite)'}">
          ${d.used_fallback ? '⚡ Zero-shot CLIP' : '🎯 Fine-tuned Model'}
        </span>
      </div>
    `;

    // Confidence bar
    html += `
      <div class="conf-section">
        <div class="conf-header">
          <span>Model Confidence</span>
          <strong>${conf.toFixed(1)}%</strong>
        </div>
        <div class="conf-track">
          <div class="conf-fill" style="width:${conf}%; background:${confColor(conf)}"></div>
        </div>
      </div>
    `;

    // Disease / healthy box
    if (!d.is_healthy && d.disease_info) {
      const info = d.disease_info;
      html += `
        <div class="disease-box" style="background:${sc.bg};border-color:${sc.border};border-left-color:${sc.text}">
          <div class="db-row">
            <strong>Cause:</strong>
            <span style="color:var(--text)"> ${info.cause}</span>
            <span class="sev-pill" style="background:${sc.pill};color:${sc.text}">${sev}</span>
          </div>
          <div class="db-row" style="margin-top:8px">
            <strong>Treatment:</strong>
            <div style="color:var(--text);margin-top:4px">${info.treatment}</div>
          </div>
        </div>
      `;
    } else if (d.is_healthy) {
      html += `<div class="healthy-box">✅ This plant appears healthy — no disease detected.<br>
        Continue regular monitoring and maintain good growing conditions.</div>`;
    }

    // Top 5 fine-tuned predictions
    html += `<div class="top5-section">
      <div class="section-label">Top 5 — Fine-tuned Model</div>`;
    d.top5.forEach((p, i) => {
      html += `
        <div class="top5-row">
          <span class="top5-rank">${i+1}</span>
          <span class="top5-name" title="${p.label}">${p.label}</span>
          <div class="bar-track">
            <div class="bar-fill" style="width:${Math.min(p.confidence,100)}%"></div>
          </div>
          <span class="top5-pct">${p.confidence.toFixed(1)}%</span>
        </div>`;
    });
    html += `</div>`;

    // Zero-shot fallback results (only if triggered)
    if (d.used_fallback && d.zeroshot_results.length > 0) {
      html += `
        <div class="fallback-section">
          <div class="fallback-title">
            ⚡ Zero-Shot CLIP Analysis
            <span style="font-weight:400;font-size:10px;color:#8a8030">
              (triggered because confidence ${conf.toFixed(1)}% &lt; ${d.threshold_used}% threshold)
            </span>
          </div>`;
      d.zeroshot_results.forEach((r, i) => {
        html += `
          <div class="fallback-row">
            <span class="top5-rank" style="color:#8a8030">${i+1}</span>
            <span class="fallback-desc" title="${r.description}">${r.description}</span>
            <span class="fallback-pct">${r.confidence.toFixed(1)}%</span>
          </div>`;
      });
      html += `</div>`;
    }

    panel.innerHTML       = html;
    panel.style.display   = 'block';
    document.getElementById('empty-state').style.display = 'none';
  }
</script>
</body>
</html>"""


# ══════════════════════════════════════════════════════════════════════════════
# FLASK ROUTES
# ══════════════════════════════════════════════════════════════════════════════
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024   # 32 MB max

ALLOWED_TYPES = {
    'image/jpeg', 'image/jpg', 'image/png', 'image/bmp',
    'image/webp', 'image/tiff', 'image/gif'
}

@app.route('/')
def index():
    return render_template_string(HTML, threshold=int(FALLBACK_THRESHOLD))

@app.route('/predict', methods=['POST'])
def predict_route():
    if 'image' not in request.files:
        return jsonify({"error": "No image in request"}), 400

    file      = request.files['image']
    threshold = float(request.form.get('threshold', FALLBACK_THRESHOLD))

    try:
        img_bytes = file.read()
        pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Resize very large images
        w, h = pil_img.size
        if max(w, h) > MAX_IMAGE_SIZE:
            ratio   = MAX_IMAGE_SIZE / max(w, h)
            pil_img = pil_img.resize((int(w*ratio), int(h*ratio)), Image.LANCZOS)

        result = predict(pil_img, threshold=threshold)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import socket
    try:
        local_ip = socket.gethostbyname(socket.gethostname())
    except Exception:
        local_ip = "127.0.0.1"

    print(f"\n{'='*55}")
    print(f"  🌿 Plant Health Detector — Ready")
    print(f"{'='*55}")
    print(f"  Local:      http://localhost:5000")
    print(f"  Network:    http://{local_ip}:5000")
    print(f"  Threshold:  {FALLBACK_THRESHOLD}% (adjustable in UI)")
    print(f"  Fallback:   Zero-shot CLIP")
    print(f"{'='*55}\n")
    app.run(host='0.0.0.0', port=5000, debug=False)