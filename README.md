# Audio–Text Sentiment Analysis System

A multimodal sentiment analysis system that combines **text** and **audio** to classify emotional tone, using a **Cross-modal Attention** mechanism to let the two modalities inform each other.

## Dataset
- **CMU-MOSEI** — a large-scale multimodal sentiment/emotion dataset
- **Text features:** timestamped words encoded with a BERT tokenizer
- **Audio features:** COVAREP acoustic features (74-dim), capturing pitch, intensity, and voice quality
- **Labels:** 3-class sentiment (Negative / Neutral / Positive); a 5-class variant (Very Negative → Very Positive) was also experimented with

## Models Implemented (for comparison)
1. **SVM Baseline** — TF-IDF (1–2 grams) + LinearSVC
2. **Text Model** — fine-tuned `bert-base-uncased` (only the last 6 Transformer layers unfrozen, to save compute and reduce overfitting)
3. **Audio Model** — Transformer Encoder (positional encoding, multi-head self-attention, feed-forward network, global average pooling) instead of a traditional RNN/BiLSTM
4. **Fusion Model (main architecture)** — symmetric cross-modal attention (Text→Audio and Audio→Text), followed by self-attention and a multi-layer classification head (Linear + BatchNorm + ReLU + Dropout)

## Results (3-class, test set)
| Model | Accuracy | F1-score |
|---|---|---|
| **Fusion (Advanced)** | **66.54%** | **66.38%** |
| Text-only (BERT) | 65.31% | 65.07% |
| SVM baseline | 56.71% | 56.70% |
| Audio-only | 48.95% | 48.51% |

The fusion model outperforms every single-modality model, showing the benefit of combining text and audio.

## Technical Improvements Applied
- Fixed a data-leakage bug (audio normalization computed only on the training set)
- Tuned `MAX_AUDIO_LEN = 600` based on sequence-length distribution analysis
- Simplified the fusion architecture to reduce overfitting
- Added stronger regularization (weight decay, dropout, label smoothing, gradient clipping)
- Early stopping and reduce-LR-on-plateau scheduling

## Web Demo
A simple web interface serves the trained fusion model for real-time prediction.

**Tech stack:** FastAPI, Uvicorn, PyTorch, Transformers, Librosa, scikit-learn

**Run locally:**
```bash
pip install -r requirements.txt
pip install librosa
python -m uvicorn app:app --reload
```
Then open `http://localhost:8000`.

## Project Structure
- `CuoikyNLP_Nhom5.ipynb` — main notebook: data pipeline, model training (SVM, Text, Audio, Fusion), evaluation
- `SoSanh_NLP_5Nhãn.ipynb` — extra experiment with 5-class sentiment labels
- `SoSanh_NLP_ThamSo.ipynb` — hyperparameter comparison experiments for the fusion model
- `sentiment_web/` — FastAPI web app serving the trained model (`app.py`, `static/index.html`)

## Note
Trained model weights (`.pt` files) are not included in this repo due to GitHub's file size limit (100MB). Models can be retrained by running the notebooks above.
