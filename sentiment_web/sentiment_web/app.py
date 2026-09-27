"""
Backend FastAPI - Phân tích cảm xúc đa phương thức
Dùng model best_fusion_model.pt đã train sẵn

Chạy: uvicorn app:app --reload
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

import torch
import torch.nn as nn
import numpy as np
import os
import io
import warnings
warnings.filterwarnings("ignore")

from transformers import BertTokenizer, BertModel, logging as tlog
tlog.set_verbosity_error()

# ─────────────────────────────────────────────
# Kiến trúc model (phải giống hệt lúc train)
# ─────────────────────────────────────────────

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.2):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.norm(self.transformer(x))


class AdvancedFusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        total_layers = len(self.bert.encoder.layer)
        for i, layer in enumerate(self.bert.encoder.layer):
            if i < (total_layers - 10):
                for param in layer.parameters():
                    param.requires_grad = False

        self.audio_proj = nn.Linear(74, 128)
        self.audio_pos  = nn.Parameter(torch.randn(1, 600, 128) * 0.1)
        self.audio_transformer = TransformerEncoder(128, 8, 3, 512, 0.2)

        self.text_proj = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 128)
        )

        self.cross_attn_t2a = nn.MultiheadAttention(128, 8, dropout=0.1, batch_first=True)
        self.cross_attn_a2t = nn.MultiheadAttention(128, 8, dropout=0.1, batch_first=True)
        self.norm_t = nn.LayerNorm(128)
        self.norm_a = nn.LayerNorm(128)
        self.self_attn = nn.MultiheadAttention(128, 4, dropout=0.1, batch_first=True)

        self.fusion = nn.Sequential(
            nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),  nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 3)
        )

    def forward(self, input_ids, attention_mask, audio):
        bert_out   = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_feat  = bert_out.last_hidden_state
        text_proj  = self.text_proj(text_feat)

        audio_feat = self.audio_proj(audio) + self.audio_pos[:, :audio.size(1), :]
        audio_feat = self.audio_transformer(audio_feat)

        attn_t2a, _ = self.cross_attn_t2a(query=text_proj, key=audio_feat, value=audio_feat)
        attn_a2t, _ = self.cross_attn_a2t(query=audio_feat, key=text_proj, value=text_proj)

        fused_t = self.norm_t(text_proj + attn_t2a)
        fused_a = self.norm_a(audio_feat + attn_a2t)

        fused_combined = torch.cat([fused_t, fused_a], dim=1)
        fused_combined, _ = self.self_attn(fused_combined, fused_combined, fused_combined)

        fused_t = fused_t.mean(dim=1)
        fused_a = fused_a.mean(dim=1)
        fused   = torch.cat([fused_t, fused_a], dim=1)
        return self.fusion(fused)


# ─────────────────────────────────────────────
# Khởi động app
# ─────────────────────────────────────────────

app = FastAPI(title="Sentiment Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = None
model     = None

MAX_AUDIO_LEN = 600
NUM_FEATURES  = 74
LABELS        = ["Negative", "Neutral", "Positive"]


def load_model():
    global tokenizer, model
    print("Đang tải tokenizer BERT...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    print("Đang tải model...")
    model = AdvancedFusionModel().to(device)

    model_path = "best_fusion_model.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Không tìm thấy '{model_path}'. "
            "Hãy đặt file best_fusion_model.pt cùng thư mục với app.py"
        )

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"✅ Model đã tải xong! Device: {device}")


@app.on_event("startup")
async def startup():
    load_model()


# Nếu có thư mục static (file HTML) thì serve luôn
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

    @app.get("/")
    def root():
        return FileResponse("static/index.html")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def make_dummy_audio() -> torch.Tensor:
    """Tạo audio zero khi chỉ có text (không có file âm thanh)."""
    audio = np.zeros((MAX_AUDIO_LEN, NUM_FEATURES), dtype=np.float32)
    return torch.tensor(audio).unsqueeze(0).to(device)  # (1, T, 74)


def preprocess_audio_bytes(audio_bytes: bytes) -> torch.Tensor:
    """
    Chuyển file audio thô thành COVAREP-like features (74 chiều).
    Vì COVAREP cần Matlab toolkit, ở đây ta dùng librosa để trích xuất
    các đặc trưng thay thế (MFCC 13 + delta 13 + delta2 13 = 39 + thêm 35 zero-pad → 74).
    """
    try:
        import librosa
        audio_np, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)

        mfcc        = librosa.feature.mfcc(y=audio_np, sr=sr, n_mfcc=13).T          # (T, 13)
        delta       = librosa.feature.delta(mfcc.T).T                                # (T, 13)
        delta2      = librosa.feature.delta(mfcc.T, order=2).T                       # (T, 13)
        features    = np.concatenate([mfcc, delta, delta2], axis=1)                  # (T, 39)

        # Pad đến 74 chiều để khớp với model
        pad_cols = NUM_FEATURES - features.shape[1]
        features = np.pad(features, ((0, 0), (0, pad_cols)), mode="constant")        # (T, 74)

        # Pad/truncate theo time axis
        T = features.shape[0]
        if T > MAX_AUDIO_LEN:
            features = features[:MAX_AUDIO_LEN]
        else:
            features = np.pad(features, ((0, MAX_AUDIO_LEN - T), (0, 0)), mode="constant")

        features = features.astype(np.float32)
        # Chuẩn hoá đơn giản (thay vì StandardScaler fit trên CMU-MOSEI)
        mean = features.mean(axis=0, keepdims=True)
        std  = features.std(axis=0, keepdims=True) + 1e-8
        features = (features - mean) / std

        return torch.tensor(features).unsqueeze(0).to(device)  # (1, 600, 74)

    except ImportError:
        # Nếu chưa cài librosa thì dùng audio giả
        return make_dummy_audio()
    except Exception:
        return make_dummy_audio()


def run_inference(text: str, audio_tensor: torch.Tensor):
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_attention_mask=True,
        return_tensors="pt"
    )
    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask, audio_tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    pred_idx    = int(np.argmax(probs))
    label       = LABELS[pred_idx]
    scores      = {LABELS[i]: float(round(probs[i], 4)) for i in range(3)}
    confidence  = float(round(float(probs[pred_idx]), 4))
    return label, scores, confidence


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

class TextRequest(BaseModel):
    text: str


@app.post("/predict/text")
def predict_text(req: TextRequest):
    """Phân tích cảm xúc chỉ từ văn bản (audio = zeros)."""
    if not req.text.strip():
        raise HTTPException(400, "Text không được để trống")

    audio_tensor        = make_dummy_audio()
    label, scores, conf = run_inference(req.text, audio_tensor)
    return {"label": label, "scores": scores, "confidence": conf, "mode": "text"}


@app.post("/predict/audio-text")
async def predict_audio_text(
    text:  str        = "",
    audio: UploadFile = File(None)
):
    """Phân tích cảm xúc từ âm thanh + văn bản (transcript)."""
    if audio is not None:
        audio_bytes  = await audio.read()
        audio_tensor = preprocess_audio_bytes(audio_bytes)
    else:
        audio_tensor = make_dummy_audio()

    text_input = text.strip() if text.strip() else "no transcript available"
    label, scores, conf = run_inference(text_input, audio_tensor)
    return {"label": label, "scores": scores, "confidence": conf, "mode": "audio+text"}


@app.get("/health")
def health():
    return {"status": "ok", "device": str(device), "model_loaded": model is not None}
