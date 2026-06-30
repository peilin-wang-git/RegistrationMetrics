"""Publication-style violin plots for registration metrics."""
from __future__ import annotations
import logging
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
LOGGER=logging.getLogger("registration_metrics")
DEFAULT_METRICS=["nmi_warped_fixed","ssim_warped_fixed","lcc_warped_fixed","dice_foreground_warped_fixed","iou_foreground_warped_fixed","hd95_foreground_warped_fixed","assd_foreground_warped_fixed","folding_ratio","jacobian_mean","VertebraNCC_warped_fixed","MovementError","MovementError_AP","MovementError_RL","MovementError_SI","MotionPCC_AllDirections","MotionAMD_AllDirections","MotionMAPE_percent_AllDirections","MotionRMSE_AllDirections","AmplitudeAMD"]

def plot_violin(metrics_csv: str|Path|None, case_motion_csv: str|Path|None, output_dir: str|Path, hue: str="center", x: str="modality", metrics: list[str]|None=None) -> None:
    """Read metric CSV files and save one PNG/PDF/SVG violin plot per metric at 600 DPI."""
    out=Path(output_dir); out.mkdir(parents=True, exist_ok=True); frames=[]
    for p in [metrics_csv, case_motion_csv]:
      if p: LOGGER.info("[PLOT] input csv=%s", p); frames.append(pd.read_csv(p))
    df=pd.concat(frames, ignore_index=True, sort=False); df.columns=[c.lower() if c in ["Center","Modality","Method"] else c for c in df.columns]
    mets=metrics or DEFAULT_METRICS; LOGGER.info("[PLOT] metrics to plot=%s", mets); LOGGER.info("[PLOT] hue=%s, x=%s", hue, x)
    plt.rcParams["font.family"]="Times New Roman"; sns.set_theme(style="whitegrid")
    for m in mets:
      if m not in df.columns: continue
      sub=df[[x,hue,m]].dropna(); LOGGER.info("[PLOT] metric=%s, valid rows=%s", m, len(sub))
      if sub.empty: continue
      fig, ax=plt.subplots(figsize=(6,4)); sns.violinplot(data=sub, x=x, y=m, hue=hue, inner="box", cut=0, ax=ax); ax.set_title(m); fig.tight_layout()
      for ext in ["png","pdf","svg"]:
        path=out/f"{m}.{ext}"; fig.savefig(path, dpi=600); LOGGER.info("[PLOT] save path=%s", path)
      plt.close(fig)
