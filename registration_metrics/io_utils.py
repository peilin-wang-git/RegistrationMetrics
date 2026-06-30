"""CSV normalization, frame extraction, and compute orchestration helpers."""
from __future__ import annotations
import logging, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd
from .config import REQUIRED_OUTPUT_COLUMNS, merged_label_map
from .orientation_utils import load_nifti, get_spacing_from_affine, get_axis_code_mapping
from .image_metrics import compute_global_metrics
from .seg_metrics import compute_segmentation_metrics
from .dvf_metrics import compute_dvf_metrics
from .vertebra_metrics import compute_vertebra_ncc
from .motion_metrics import compute_organ_ncc_moves
from .gpu_utils import get_device, device_name

LOGGER=logging.getLogger("registration_metrics")
METADATA_COLUMNS={"case_id","frame","row_index","Method","Center","Modality","Task","Organ","AnalysisGroup","fixed_img_path","moving_img_path","warped_img_path","fixed_seg_path","moving_seg_path","warped_seg_path","transform_path","status","error_message","skip_reason","completed_at","run_id","runtime_seconds"}
COLUMN_ALIASES={"FixedImagePath":"fixed_img_path","MovingImagePath":"moving_img_path","WarpedImagePath":"warped_img_path","FixedSegmentationPath":"fixed_seg_path","MovingSegmentationPath":"moving_seg_path","WarpedSegmentationPath":"warped_seg_path","TransformPath":"transform_path","Frame":"frame","3DName":"fixed_img_name","4DName":"moving_img_name"}

def append_rows_to_csv(rows: list[dict], output_path: Path) -> None:
    """Append rows to CSV with a header only on first write; closing the file flushes to disk."""
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists()
    pd.DataFrame(rows).to_csv(output_path, mode="a", header=write_header, index=False)
    LOGGER.info("[SAVE PROGRESS] appended_rows=%s path=%s", len(rows), output_path)

def save_summary_from_progress(metrics_progress_path: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reload progress CSV from disk and save grouped/overall count, mean, variance, std, median summaries."""
    LOGGER.info("[SUMMARY] reloading progress csv from %s", metrics_progress_path)
    if not metrics_progress_path.exists() or metrics_progress_path.stat().st_size == 0:
        raise FileNotFoundError(f"[SUMMARY] progress CSV missing or empty: {metrics_progress_path}")
    df = pd.read_csv(metrics_progress_path)
    LOGGER.info("[SUMMARY] loaded rows=%s, columns=%s", len(df), list(df.columns))
    numeric = [c for c in df.select_dtypes(include="number").columns if c not in METADATA_COLUMNS]
    LOGGER.info("[SUMMARY] numeric metric columns=%s variance_ddof=1", numeric)
    group_cols = [c for c in ["Method", "Center", "Modality", "Task", "Organ"] if c in df.columns]
    agg = ["count", "mean", "var", "std", "median"]
    group_summary = df.groupby(group_cols, dropna=False)[numeric].agg(agg).reset_index() if numeric and group_cols else pd.DataFrame()
    overall = df[numeric].agg(agg).T.reset_index().rename(columns={"index": "metric"}) if numeric else pd.DataFrame()
    gp = output_dir / "summary_by_group.csv"; op = output_dir / "summary_overall.csv"
    group_summary.to_csv(gp, index=False); overall.to_csv(op, index=False)
    LOGGER.info("[SUMMARY] saved group summary to %s", gp); LOGGER.info("[SUMMARY] saved overall summary to %s", op)
    return group_summary, overall

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize old and new CSV columns to snake_case names."""
    out=df.rename(columns=COLUMN_ALIASES).copy()
    out.columns=[c if c in COLUMN_ALIASES.values() else ''.join(['_'+x.lower() if x.isupper() else x for x in c]).lstrip('_') for c in out.columns]
    return out

def _frame(data: np.ndarray, idx: int) -> np.ndarray:
    """Return 3D frame; 3D arrays are reused for all frames."""
    return data if data.ndim == 3 else data[..., idx]

def compute_from_config(config: dict, output_dir: str|Path, enable_global=True, enable_seg=True, enable_dvf=True, enable_motion=True, enable_vertebra=True, use_gpu: bool = False, requested_device: str = "cuda:0", gpu_metrics: str = "all", ncc_batch_size: int = 64) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    """Compute metrics for all configured method/group CSV files with case-level error isolation."""
    outdir=Path(output_dir); outdir.mkdir(parents=True, exist_ok=True); labels=merged_label_map(config); all_rows=[]; errors=[]; device=get_device(use_gpu, requested_device); progress_path=outdir/"detailed_progress.csv"; error_path=outdir/"error_log.csv"
    LOGGER.info("[CONFIG] methods=%s output_dir=%s", list(config.keys()), outdir)
    for method, groups in config.items():
      if method == "label_map": continue
      for analysis_group, g in groups.items():
        LOGGER.info("[GROUP START] input_csv=%s method=%s group=%s center=%s modality=%s task=%s organ=%s device=%s", g.get("csv_path"), method, analysis_group, g.get('center'), g.get('modality'), g.get('task'), g.get('organ'), device_name(device))
        df=normalize_columns(pd.read_csv(g["csv_path"])); group_rows=[]
        total_rows=len(df)
        for row_index, r in df.iterrows():
          t0=time.time(); row_results=[]; case=str(r.get("case_id", r.get("run_id", len(group_rows))))
          LOGGER.info("[CSV ROW START] input_csv=%s row=%s/%s case_id=%s method=%s center=%s modality=%s task=%s organ=%s", g.get("csv_path"), row_index + 1, total_rows, case, method, g.get("center"), g.get("modality"), g.get("task"), g.get("organ"))
          LOGGER.info("[CASE START] method=%s center=%s modality=%s organ=%s case_id=%s", method, g.get('center'), g.get('modality'), g.get('organ'), case)
          base={"case_id":case,"fixed_img_path":r.get("fixed_img_path"),"moving_img_path":r.get("moving_img_path"),"warped_img_path":r.get("warped_img_path"),"fixed_seg_path":r.get("fixed_seg_path"),"moving_seg_path":r.get("moving_seg_path"),"warped_seg_path":r.get("warped_seg_path"),"transform_path":r.get("transform_path"),"Method":method,"Center":g.get("center"),"Modality":g.get("modality"),"Task":g.get("task"),"Organ":g.get("organ"),"AnalysisGroup":analysis_group,"row_index":row_index,"status":"ok","error_message":"","skip_reason":""}
          try:
            imgs={k: load_nifti(r[k]) for k in ["fixed_img_path","moving_img_path","warped_img_path"] if pd.notna(r.get(k))}
            segs={k: load_nifti(r[k]) for k in ["fixed_seg_path","moving_seg_path","warped_seg_path"] if pd.notna(r.get(k))}
            get_axis_code_mapping(imgs["fixed_img_path"]); spacing=get_spacing_from_affine(imgs["fixed_img_path"])
            data={k: np.asanyarray(v.dataobj) for k,v in imgs.items()}; sdata={k: np.asanyarray(v.dataobj) for k,v in segs.items()}
            nframes=max([data[k].shape[3] if data[k].ndim==4 else 1 for k in data]); LOGGER.info("[FRAME] case=%s, total_frames=%s", case, nframes)
            for frame in range(nframes):
              LOGGER.info("[FRAME START] case_id=%s row=%s frame=%s/%s method=%s center=%s modality=%s task=%s organ=%s device=%s", case, row_index, frame+1, nframes, method, g.get("center"), g.get("modality"), g.get("task"), g.get("organ"), device_name(device))
              fixed=_frame(data["fixed_img_path"], frame); moving=_frame(data["moving_img_path"], frame); warped=_frame(data["warped_img_path"], frame)
              fseg=_frame(sdata["fixed_seg_path"], min(frame, sdata["fixed_seg_path"].shape[3]-1) if sdata["fixed_seg_path"].ndim==4 else 0)
              mseg=_frame(sdata["moving_seg_path"], min(frame, sdata["moving_seg_path"].shape[3]-1) if sdata["moving_seg_path"].ndim==4 else 0)
              wseg=_frame(sdata["warped_seg_path"], min(frame, sdata["warped_seg_path"].shape[3]-1) if sdata["warped_seg_path"].ndim==4 else 0)
              LOGGER.info("[FRAME] image shapes fixed/moving/warped=%s/%s/%s", fixed.shape,moving.shape,warped.shape); LOGGER.info("[FRAME] seg shapes fixed/moving/warped=%s/%s/%s", fseg.shape,mseg.shape,wseg.shape)
              row=base.copy(); row["Frame"]=frame
              if fixed.shape != moving.shape or fixed.shape != warped.shape: row.update(status="skipped", skip_reason="shape_mismatch")
              else:
                if enable_global: row.update(compute_global_metrics(fixed,moving,warped,case,frame,row_index=row_index,device=device))
                if enable_seg: row.update(compute_segmentation_metrics(fseg,mseg,wseg,labels,spacing,case,frame,row_index=row_index,device=device))
                if enable_motion: row.update(compute_organ_ncc_moves(fixed,moving,warped,fseg,mseg,wseg,labels,imgs["fixed_img_path"].affine,case,frame,row_index=row_index,device=device,ncc_batch_size=ncc_batch_size))
                if enable_vertebra: row.update(compute_vertebra_ncc(fixed,moving,warped,fseg,mseg,wseg,case,frame,row_index=row_index,device=device))
              row["runtime_seconds"]=time.time()-t0; all_rows.append(row); group_rows.append(row); row_results.append(row)
            if enable_dvf and pd.notna(r.get("transform_path")): row_results[-1].update(compute_dvf_metrics(r["transform_path"], case, row_index=row_index, device=device)); all_rows[-1].update(row_results[-1])
            append_rows_to_csv(row_results, progress_path); LOGGER.info("[SAVE PROGRESS] completed input row %s/%s case_id=%s", row_index + 1, total_rows, case)
          except (OSError, ValueError, KeyError, RuntimeError) as e:
            msg=f"{type(e).__name__}: {e}"; LOGGER.error("[ERROR] case=%s error=%s traceback=%s", case, msg, traceback.format_exc(limit=3)); er=base|{"Frame":r.get("frame",0),"status":"error","error_message":msg,"runtime_seconds":time.time()-t0}; all_rows.append(er); errors.append(er); append_rows_to_csv([er], error_path); append_rows_to_csv([er], progress_path); LOGGER.info("[SAVE PROGRESS] completed input row %s/%s case_id=%s", row_index + 1, total_rows, case)
        p=outdir/f"metrics_{method}_{g.get('center')}_{g.get('modality')}_{g.get('organ')}.csv".replace('/','_').replace(' ','_'); pd.DataFrame(group_rows).to_csv(p,index=False); LOGGER.info("[SAVE] group detailed CSV=%s", p)
    combined=pd.DataFrame(all_rows)
    for c in REQUIRED_OUTPUT_COLUMNS:
      if c not in combined: combined[c]=np.nan
    combined.to_csv(outdir/"combined_metrics.csv", index=False); LOGGER.info("[SAVE] combined_metrics.csv=%s", outdir/"combined_metrics.csv")
    pd.DataFrame(errors).to_csv(outdir/"error_log.csv", index=False); LOGGER.info("[SAVE] error_log.csv=%s", outdir/"error_log.csv")
    save_summary_from_progress(progress_path, outdir)
    return combined, pd.DataFrame(), pd.DataFrame()
