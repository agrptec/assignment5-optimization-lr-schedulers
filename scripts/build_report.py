"""Render a concise PDF report from the notebook's computed results."""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.image import imread
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REPORT = ROOT / "report" / "Assignment_5_Report.pdf"
summary = pd.read_csv(RESULTS / "summary.csv")
fig, ax = plt.subplots(figsize=(8.27, 11.69))
ax.axis("off")
fig.text(0.08, 0.95, "Assignment 5 — Optimization and Learning Rate Schedulers", fontsize=15, weight="bold")
fig.text(0.08, 0.915, "Raisin dataset · 7–20–3–1 network · L1 λ = 0.7", fontsize=10)
fig.text(0.08, 0.88, "Setup", fontsize=12, weight="bold")
fig.text(0.08, 0.855, "900 observations; stratified 60/20/20 split; train-only standardization.\n"
         "5,000 epochs, batch size 64, identical initialization and batch order.\n"
         "Initial LR: 0.1 for GD/Momentum and 0.01 for Adam.", fontsize=9.4, va="top")
fig.text(0.08, 0.76, "Final results", fontsize=12, weight="bold")
columns = ["Optimizer", "Scheduler", "Val. loss", "Val. accuracy"]
rows = [[r.optimizer.upper(), r.scheduler.replace("tutorial_decay", "Tutorial decay"),
         f"{r.validation_loss:.4f}", f"{r.validation_accuracy_pct:.2f}%"]
        for r in summary.itertuples()]
table = ax.table(cellText=rows, colLabels=columns, cellLoc="left", colLoc="left",
                 bbox=(0.05, 0.39, 0.9, 0.34), colWidths=[0.18, 0.36, 0.2, 0.26])
table.auto_set_font_size(False)
table.set_fontsize(8.4)
for (row, _), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor("#dfe9f3")
        cell.set_text_props(weight="bold")
    elif row % 2 == 0:
        cell.set_facecolor("#f6f8fa")
fig.text(0.08, 0.35, "Discussion", fontsize=12, weight="bold")
fig.text(0.08, 0.325,
         "Tutorial decay lowered validation loss for GD (0.5821 → 0.5534) and Momentum\n"
         "(0.5706 → 0.5588), but raised it for Adam (0.5515 → 0.5774). Step decay\n"
         "gave GD and Momentum 86.67% validation accuracy. Adam with a constant\n"
         "rate had the lowest validation loss; its held-out test accuracy was 87.22%.\n"
         "The training/validation loss gap (about 0.26 vs 0.55–0.58) suggests overfitting.\n"
         "Accuracy and cross-entropy rank runs differently because confidence matters.",
         fontsize=9.2, va="top", linespacing=1.6)
fig.text(0.08, 0.21, "We selected by validation loss and used the test set once; a single split cannot establish a universal winner.",
         fontsize=8.8)
fig.text(0.08, 0.16, "The tutorial's 'exponential' formula is inverse-time: LR₀/(1 + 0.001 × epoch).",
         fontsize=8.5)
fig.text(0.08, 0.13, "Step decay: LR₀ × 0.7^(floor(epoch / 1000)). Selection used validation loss only.",
         fontsize=8.5)
fig.text(0.08, 0.085, "Data: Çinar, Koklu & Tasdemir (2020), UCI Raisin, DOI: 10.24432/C5660T (CC BY 4.0).",
         fontsize=7.6)
fig.text(0.08, 0.065, "Optimizer background: Kingma & Ba (2014), Adam. Course Optimization Methods notebook.",
         fontsize=7.6)
with PdfPages(REPORT) as pdf:
    pdf.savefig(fig)
    plt.close(fig)
    for title, filename in [("Training and validation loss", "loss_curves.png"),
                            ("Learning rate schedules", "learning_rates.png")]:
        image = imread(RESULTS / filename)
        page, axis = plt.subplots(figsize=(11.69, 8.27))
        axis.imshow(image)
        axis.axis("off")
        page.suptitle(title, fontsize=14)
        pdf.savefig(page)
        plt.close(page)
print(REPORT)
