# 🕵️‍♂️ Where's Waldo? — Using Deep Learning to Find Waldo

> ### Can a Computer Answer *"Where's Waldo?"*
> ### 🎯 Using AI (YOLO) to Find Waldo in a Sea of Faces

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kaopanboonyuen/WALDO2026/blob/main/code/WALDO_AI_NSTDA2026_toStudent.ipynb)
![Runs on Colab T4 GPU](https://img.shields.io/badge/Colab-T4%20GPU%20Free-orange?logo=googlecolab)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv8%20%7C%20YOLO11%20%7C%20YOLO12-blueviolet)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Made with ❤️](https://img.shields.io/badge/Made%20with-%E2%9D%A4%EF%B8%8F-red)
![NSTDA](https://img.shields.io/badge/NSTDA-AI%20Workshop%202026-1f6feb)

**AI Lab — architected by Dr. Teerapong Panboonyuen (P'Kao)**

---

> 🎯 **Goal**: Somewhere in a crowd of thousands of tiny cartoon people, **Waldo** is hiding. Instead of scanning the page for ten minutes with your own eyes, this workshop teaches you to train an **AI object detector (YOLO)** to find him in **milliseconds** — while learning how modern AI has evolved from classical computer vision → deep learning → LLMs → today's agentic, multimodal AI, and *why the biggest model isn't always the right tool for the job.*

---

## 🧭 Table of Contents

- [Overview](#-overview)
- [In-Class Lab: Where's Waldo](#️️-in-class-lab-wheres-waldo)
- [What You'll Learn](#-what-youll-learn)
- [Repo Structure](#-repo-structure)
- [Quickstart](#-quickstart)
- [The Model Zoo](#-the-model-zoo)
- [Evaluation & the Scoreboard](#-evaluation--the-scoreboard)
- [Mini-Hackathon Assignment](#-mini-hackathon-homework-assignment)
- [Instructor](#-instructor)
- [Acknowledgements](#-acknowledgements)

---

## 📖 Overview

This repository is the full teaching kit for the **NSTDA AI Workshop 2026** session:

> **"Can a Computer Answer 'Where's Waldo?'" — Using Deep Learning to Find Waldo**

It walks students from **zero to a working object detector** — and then challenges them to beat it.

| Part | Topic |
|:---:|---|
| 🧠 **Part 1** | The AI Landscape 2026 — classical CV → deep learning → LLMs → agentic & multimodal AI, plus AI ethics & governance |
| 🔍 **Part 2** | Building a Waldo Detector — dataset, EDA, training 3 generations of YOLO, evaluation, and a live Scoreboard |
| 🛠️ **Part 3** | Fine-Tuning Like a Pro — epochs, learning rate, loss weighting, class imbalance, and a competition playbook |
| 🏆 **Homework** | Mini-Hackathon — train your own 4th model and try to beat the class Scoreboard |

---

## 🕵️‍♂️ In-Class Lab: Where's Waldo

| Resource | Link |
|:---|:---|
| 🧠 Lecture Slide | [`slides/From_Pixels_to_Waldo.pdf`](slides/From_Pixels_to_Waldo.pdf) |
| 🧪 Colab Notebook | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kaopanboonyuen/WALDO2026/blob/main/code/WALDO_AI_NSTDA2026_toStudent.ipynb) |
| 📓 Notebook (source) | [`code/WALDO_AI_NSTDA2026_toStudent.ipynb`](code/WALDO_AI_NSTDA2026_toStudent.ipynb) |
| 📂 Dataset | Where's Waldo YOLO-format dataset (`waldo` · `wenda` classes) — downloaded automatically inside the notebook |

> 💡 **Lab Topics**
>
> - Explore a real "Where's Waldo?" image dataset, understand the YOLO label format, and run full **EDA** (class balance, bounding-box size, resolution, visual overlays) before training anything.
> - Train **three generations of YOLO** (YOLOv8n → YOLO11n → YOLOv12n) on the exact same data, then add a **4th model of your own**.
> - Build a **live Scoreboard** (Precision · Recall · F1 · mAP50 · mAP50-95 · speed · size) and crown a **Winner** with a weighted composite score.
> - Visualize results five different ways: accuracy bars, training curves, a radar chart, confusion matrices, and a speed-vs-accuracy bubble chart.
> - Every run is dynamically resolved from the `runs/` folder — **no hardcoded `train`, `train2`, `train3`** paths. Re-run any cell as many times as you like.

---

## 🧠 What You'll Learn

<table>
<tr>
<td width="50%" valign="top">

### 🌐 Part 1 — The AI Landscape, 2026
- 🖼️ Classical computer vision (edges, HOG/SIFT, Haar cascades)
- 🧬 The Deep Learning revolution (CNNs, ResNet, end-to-end learning)
- 🗣️ The LLM era (transformers, attention, chat, reasoning)
- 🤖 2026 vocabulary: **AI Agent**, **Agentic AI**, **Multimodal**, **Multi-Model**
- ⚖️ AI Ethics & Governance: fairness, privacy, transparency, regulation
- 💡 *Big idea:* you don't need the biggest model to do a small, well-defined job

</td>
<td width="50%" valign="top">

### 🔍 Part 2 — Building a Waldo Detector
- 📦 The YOLO dataset format (`images/` + `labels/` + `data.yaml`)
- 📊 EDA first: class balance, box size, resolution, sample overlays
- 🏋️ Training pipeline: `data.yaml` → `YOLO(weights)` → `model.train()` → `best.pt` → `model.val()`
- 📏 Evaluation metrics: Precision, Recall, F1, mAP50, mAP50-95
- 🏆 The Scoreboard & composite score, with 5 comparison plots

</td>
</tr>
</table>

<table>
<tr>
<td width="50%" valign="top">

### 🛠️ Part 3 — Fine-Tuning Like a Pro
- ⏱️ `epochs`, `lr0` / `lrf`, `patience`, `cos_lr`
- 🎛️ The nerdy-but-runnable cheat sheet: `imgsz`, `optimizer`, `mosaic`, `mixup`, `copy_paste`, `warmup_epochs`
- ⚖️ Handling class imbalance (when Waldo is rarer than Wenda)
- 🧮 What's inside the YOLO loss: `box` (CIoU) · `cls` (BCE) · `dfl` (Distribution Focal Loss)
- 🚀 Competition & production playbook: TTA, ensembling, export (ONNX/TensorRT/TFLite)

</td>
<td width="50%" valign="top">

### 🎓 Homework — Mini-Hackathon
- Train your own YOLO experiment(s) on the **same dataset**
- Beat today's best composite score
- Any `yolov8*` / `yolo11*` / `yolo12*` size or config
- Graded on correctness, improvement, reasoning, and code cleanliness

</td>
</tr>
</table>

---

## 📂 Repo Structure

```

WALDO2026/
├── slides/
│   └── From_Pixels_to_Waldo.pdf        # Full lecture deck (Parts 1–3 + homework)
├── code/
│   └── WALDO_AI_NSTDA2026_toStudent.ipynb   # The hands-on Colab notebook
└── README.md

```

---

## 🚀 Quickstart

1. Click the **Open in Colab** badge above (or open [`code/WALDO_AI_NSTDA2026_toStudent.ipynb`](code/WALDO_AI_NSTDA2026_toStudent.ipynb) directly).
2. In Colab: `Runtime → Change runtime type → T4 GPU` (free tier works fine).
3. Run the cells top to bottom:
   - 🔧 Setup → 📦 Dataset → 📊 EDA → ⚙️ `data.yaml` → 🏋️ Train 3 models → ✏️ **your 4th model** → 🏆 Scoreboard → 📈 Visualizations → 🥇 Winner → 🖼️ Test on a new image.
4. Complete the **Mini-Hackathon** section at the end as homework.

> 🔒 *Instructors:* the notebook contains a clearly marked `SOLUTION — INSTRUCTOR ONLY` cell for the "your turn" exercise — delete that section before distributing to students.

---

## 🏗️ The Model Zoo

| Model | Released | Core Idea |
|---|:---:|---|
| `YOLOv8n` | 2023 | The battle-tested classic — our baseline |
| `YOLO11n` | 2024 | Ultralytics' redesigned, more efficient successor |
| `YOLOv12n` | 2025 | Attention-centric architecture — the cutting edge |
| **Your model** | 🫵 | Whatever you build in the "your turn" exercise |

Same data, same epochs, same image size → the comparison stays fair.

---

## 📏 Evaluation & the Scoreboard

Every model is scored on:

`Precision` · `Recall` · `F1` · `mAP50` · `mAP50-95` · inference speed · model size

combined into a weighted **Composite Score**:

```

0.50 · mAP50-95  +  0.20 · mAP50  +  0.15 · F1  +  0.10 · speed_norm  +  0.05 · size_norm

```

Weighted toward accuracy — but a fast, compact model still earns real credit. 🏎️

---

## 🎓 Mini-Hackathon (Homework Assignment)

**Objective:** Build your own YOLO experiment(s) on the same dataset and push the highest composite score you can.

**Rules**
- ✅ Same `data.yaml` / dataset — no extra labeled data
- ✅ Any YOLO family/size: `yolov8*`, `yolo11*`, `yolo12*`
- ✅ Tune architecture, epochs, `imgsz`, augmentation, optimizer, LR schedule
- 🚫 No hand-labeling and no training on the test set

| Criterion | Weight |
|---|:---:|
| Trains & evaluates without errors | 30% |
| Improvement over the class's best score | 40% |
| Clarity of reasoning / EDA-informed decisions | 20% |
| Code cleanliness (reuses dynamic helpers, no hardcoded paths) | 10% |

**Ideas to try:** bigger backbone vs. more epochs on `n` · higher `imgsz` (960/1280) for tiny targets · optimizer & LR-schedule comparisons · ensembling two checkpoints.

---

## 👨‍🏫 Instructor

**Dr. Teerapong Panboonyuen (P'Kao)**
AI Lab — NSTDA AI Workshop 2026
🔗 [github.com/kaopanboonyuen/WALDO2026](https://github.com/kaopanboonyuen/WALDO2026)

---

## 🙏 Acknowledgements

- Models: [Ultralytics YOLOv8 / YOLO11 / YOLO12](https://docs.ultralytics.com/)
- Dataset & original "Where's Waldo?" concept, and workshop design: NSTDA AI Workshop 2026
- Built for the next generation of Thai 🇹🇭 AI builders

---

<p align="center">Made with 🕵️‍♂️ + ☕ — now go find Waldo (and beat the Scoreboard) 🏆</p>