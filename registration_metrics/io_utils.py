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

LOGGER=logging.getLogger("registration_metrics")
COLUMN_ALIASES={"FixedImagePath":"fixed_img_path","MovingImagePath":"moving_img_path","WarpedImagePath":"warped_img_path","FixedSegmentationPath":"fixed_seg_path","MovingSegmentationPath":"moving_seg_path","WarpedSegmentationPath":"warped_seg_path","TransformPath":"transform_path","Frame":"frame","3DName":"fixed_img_name","4DName":"moving_img_name"}

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize old and new CSV columns to snake_case names."""
    out=df.rename(columns=COLUMN_ALIASES).copy()
    out.columns=[c if c in COLUMN_ALIASES.values() else ''.join(['_'+x.lower() if x.isupper() else x for x in c]).lstrip('_') for c in out.columns]
    return out

def _frame(data: np.ndarray, idx: int) -> np.ndarray:
    """Return 3D frame; 3D arrays are reused for all frames."""
    return data if data.ndim == 3 else data[..., idx]

def compute_from_config(config: dict, output_dir: str|Path, enable_global=True, enable_seg=True, enable_dvf=True, enable_motion=True, enable_vertebra=True) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    """Compute metrics for all configured method/group CSV files with case-level error isolation."""
    outdir=Path(output_dir); outdir.mkdir(parents=True, exist_ok=True); labels=merged_label_map(config); all_rows=[]; errors=[]
    LOGGER.info("[CONFIG] methods=%s output_dir=%s", list(config.keys()), outdir)
    for method, groups in config.items():
      if method == "label_map": continue
      for analysis_group, g in groups.items():
        LOGGER.info("[GROUP START] method=%s group=%s center=%s modality=%s organ=%s", method, analysis_group, g.get('center'), g.get('modality'), g.get('organ'))
        df=normalize_columns(pd.read_csv(g["csv_path"])); group_rows=[]
        for _, r in df.iterrows():
          t0=time.time(); case=str(r.get("case_id", r.get("run_id", len(group_rows))))
          LOGGER.info("[CASE START] method=%s center=%s modality=%s organ=%s case_id=%s", method, g.get('center'), g.get('modality'), g.get('organ'), case)
          base={"case_id":case,"fixed_img_path":r.get("fixed_img_path"),"moving_img_path":r.get("moving_img_path"),"warped_img_path":r.get("warped_img_path"),"fixed_seg_path":r.get("fixed_seg_path"),"moving_seg_path":r.get("moving_seg_path"),"warped_seg_path":r.get("warped_seg_path"),"transform_path":r.get("transform_path"),"Method":method,"Center":g.get("center"),"Modality":g.get("modality"),"Task":g.get("task"),"Organ":g.get("organ"),"AnalysisGroup":analysis_group,"status":"ok","error_message":"","skip_reason":""}
          try:
            imgs={k: load_nifti(r[k]) for k in ["fixed_img_path","moving_img_path","warped_img_path"] if pd.notna(r.get(k))}
            segs={k: load_nifti(r[k]) for k in ["fixed_seg_path","moving_seg_path","warped_seg_path"] if pd.notna(r.get(k))}
            get_axis_code_mapping(imgs["fixed_img_path"]); spacing=get_spacing_from_affine(imgs["fixed_img_path"])
            data={k: np.asanyarray(v.dataobj) for k,v in imgs.items()}; sdata={k: np.asanyarray(v.dataobj) for k,v in segs.items()}
            nframes=max([data[k].shape[3] if data[k].ndim==4 else 1 for k in data]); LOGGER.info("[FRAME] case=%s, total_frames=%s", case, nframes)
            for frame in range(nframes):
              LOGGER.info("[FRAME] processing frame %s/%s", frame+1, nframes)
              fixed=_frame(data["fixed_img_path"], frame); moving=_frame(data["moving_img_path"], frame); warped=_frame(data["warped_img_path"], frame)
              fseg=_frame(sdata["fixed_seg_path"], min(frame, sdata["fixed_seg_path"].shape[3]-1) if sdata["fixed_seg_path"].ndim==4 else 0)
              mseg=_frame(sdata["moving_seg_path"], min(frame, sdata["moving_seg_path"].shape[3]-1) if sdata["moving_seg_path"].ndim==4 else 0)
              wseg=_frame(sdata["warped_seg_path"], min(frame, sdata["warped_seg_path"].shape[3]-1) if sdata["warped_seg_path"].ndim==4 else 0)
              LOGGER.info("[FRAME] image shapes fixed/moving/warped=%s/%s/%s", fixed.shape,moving.shape,warped.shape); LOGGER.info("[FRAME] seg shapes fixed/moving/warped=%s/%s/%s", fseg.shape,mseg.shape,wseg.shape)
              row=base.copy(); row["Frame"]=frame
              if fixed.shape != moving.shape or fixed.shape != warped.shape: row.update(status="skipped", skip_reason="shape_mismatch")
              else:
                if enable_global: row.update(compute_global_metrics(fixed,moving,warped,case,frame))
                if enable_seg: row.update(compute_segmentation_metrics(fseg,mseg,wseg,labels,spacing,case,frame))
                if enable_motion: row.update(compute_organ_ncc_moves(fixed,moving,warped,fseg,mseg,wseg,labels,imgs["fixed_img_path"].affine,case,frame))
                if enable_vertebra: row.update(compute_vertebra_ncc(fixed,moving,warped,fseg,mseg,wseg,case,frame))
              row["runtime_seconds"]=time.time()-t0; all_rows.append(row); group_rows.append(row)
            if enable_dvf and pd.notna(r.get("transform_path")): all_rows[-1].update(compute_dvf_metrics(r["transform_path"], case))
          except (OSError, ValueError, KeyError, RuntimeError) as e:
            msg=f"{type(e).__name__}: {e}"; LOGGER.error("[ERROR] case=%s error=%s traceback=%s", case, msg, traceback.format_exc(limit=3)); er=base|{"Frame":r.get("frame",0),"status":"error","error_message":msg,"runtime_seconds":time.time()-t0}; all_rows.append(er); errors.append(er)
        p=outdir/f"metrics_{method}_{g.get('center')}_{g.get('modality')}_{g.get('organ')}.csv".replace('/','_').replace(' ','_'); pd.DataFrame(group_rows).to_csv(p,index=False); LOGGER.info("[SAVE] group detailed CSV=%s", p)
    combined=pd.DataFrame(all_rows)
    for c in REQUIRED_OUTPUT_COLUMNS:
      if c not in combined: combined[c]=np.nan
    combined.to_csv(outdir/"combined_metrics.csv", index=False); LOGGER.info("[SAVE] combined_metrics.csv=%s", outdir/"combined_metrics.csv")
    pd.DataFrame(errors).to_csv(outdir/"error_log.csv", index=False); LOGGER.info("[SAVE] error_log.csv=%s", outdir/"error_log.csv")
    return combined, pd.DataFrame(), pd.DataFrame()
