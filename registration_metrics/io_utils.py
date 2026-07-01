"""CSV normalization and 3D-only compute orchestration helpers."""
from __future__ import annotations
import logging, time, traceback
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from .config import REQUIRED_OUTPUT_COLUMNS, merged_label_map, resolve_seg_metric_organs, resolve_seg_mean_organs, resolve_mask_quality_thresholds
from .orientation_utils import load_nifti, get_spacing_from_affine, get_axis_code_mapping
from .image_metrics import compute_global_metrics
from .seg_metrics import compute_segmentation_metrics
from .dvf_metrics import compute_dvf_metrics
from .vertebra_metrics import compute_vertebra_ncc
from .motion_metrics import compute_organ_ncc_moves
from .gpu_utils import get_device

LOGGER=logging.getLogger("registration_metrics")
METADATA_COLUMNS={"case_id","frame","row_index","Method","Center","Modality","Task","Organ","AnalysisGroup","fixed_img_path","moving_img_path","warped_img_path","fixed_seg_path","moving_seg_path","warped_seg_path","transform_path","status","error_message","skip_reason","completed_at","run_id","runtime_seconds"}
COLUMN_ALIASES={"FixedImagePath":"fixed_img_path","MovingImagePath":"moving_img_path","WarpedImagePath":"warped_img_path","FixedSegmentationPath":"fixed_seg_path","MovingSegmentationPath":"moving_seg_path","WarpedSegmentationPath":"warped_seg_path","TransformPath":"transform_path","Frame":"frame","3DName":"fixed_img_name","4DName":"moving_img_name"}

@dataclass
class CaseTask:
    method: str
    center: Any
    modality: Any
    task: Any
    organ: Any
    analysis_group: str
    csv_path: str
    row_index: int
    total_rows: int
    row_dict: dict[str, Any]
    labels: dict[int, str]
    selected_seg_organs: list[str]
    selected_seg_mean_organs: list[str]
    enable_global: bool
    enable_seg: bool
    enable_dvf: bool
    enable_motion: bool
    enable_vertebra: bool
    use_gpu: bool
    requested_device: str
    ncc_batch_size: int
    verbose_seg_mean: bool
    min_mask_volume_voxels: int
    severe_volume_ratio_threshold: float
    nmi_bins: int


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


def normalize_intensity_0_1(image: np.ndarray, image_name: str, case_id: str, row_index: int) -> np.ndarray:
    """Normalize one intensity image to [0, 1] using finite-voxel min/max."""
    x = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(x)
    finite_voxels = int(finite.sum())
    if finite_voxels == 0:
        LOGGER.warning("[INTENSITY NORM] case_id=%s row=%s image=%s reason=no_finite_voxels normalized=set_to_nan", case_id, row_index, image_name)
        return np.full(x.shape, np.nan, dtype=np.float32)
    vals = x[finite]
    raw_min = float(vals.min()); raw_max = float(vals.max()); raw_mean = float(vals.mean()); raw_std = float(vals.std())
    if raw_max == raw_min:
        LOGGER.warning("[INTENSITY NORM] case_id=%s row=%s image=%s reason=constant_image raw_min=%s raw_max=%s normalized=set_to_zeros", case_id, row_index, image_name, raw_min, raw_max)
        out = np.zeros(x.shape, dtype=np.float32)
        out[~finite] = np.nan
        return out
    out = ((x - raw_min) / (raw_max - raw_min)).astype(np.float32)
    out[~finite] = np.nan
    normalized_vals = out[np.isfinite(out)]
    LOGGER.info("[INTENSITY NORM] case_id=%s row=%s image=%s finite_voxels=%s raw_min=%s raw_max=%s raw_mean=%s raw_std=%s normalized_min=%s normalized_max=%s dtype=%s device=cpu", case_id, row_index, image_name, finite_voxels, raw_min, raw_max, raw_mean, raw_std, float(normalized_vals.min()), float(normalized_vals.max()), out.dtype)
    return out


def _frame_value(r: dict[str, Any]) -> Any:
    return r.get("frame", r.get("Frame", 0)) if pd.notna(r.get("frame", r.get("Frame", 0))) else 0


def _base_row(task: CaseTask, case: str, frame: Any, status: str = "ok", error_message: str = "", skip_reason: str = "") -> dict[str, Any]:
    r = task.row_dict
    return {"case_id":case,"fixed_img_path":r.get("fixed_img_path"),"moving_img_path":r.get("moving_img_path"),"warped_img_path":r.get("warped_img_path"),"fixed_seg_path":r.get("fixed_seg_path"),"moving_seg_path":r.get("moving_seg_path"),"warped_seg_path":r.get("warped_seg_path"),"transform_path":r.get("transform_path"),"Method":task.method,"Center":task.center,"Modality":task.modality,"Task":task.task,"Organ":task.organ,"AnalysisGroup":task.analysis_group,"row_index":task.row_index,"Frame":frame,"status":status,"error_message":error_message,"skip_reason":skip_reason}


def _non3d_skip(case: str, row_index: int, path: Any, arr: np.ndarray) -> str:
    msg = "current pipeline expects pre-split 3D cases"
    LOGGER.warning("[SKIP NON-3D] case_id=%s row=%s path=%s ndim=%s shape=%s reason=%s", case, row_index, path, arr.ndim, arr.shape, msg)
    return msg


def compute_single_case_task(task: CaseTask) -> dict[str, Any]:
    """Compute one CSV row as one pre-split 3D case and return rows for the main process to save."""
    t0=time.time(); r=task.row_dict; case=str(r.get("case_id", r.get("run_id", task.row_index))); frame=_frame_value(r); device=get_device(task.use_gpu, task.requested_device)
    base=_base_row(task, case, frame)
    LOGGER.info("[CSV ROW START] input_csv=%s row=%s/%s case_id=%s method=%s center=%s modality=%s task=%s organ=%s", task.csv_path, task.row_index + 1, task.total_rows, case, task.method, task.center, task.modality, task.task, task.organ)
    LOGGER.info("[CASE START] row=%s case_id=%s mode=3D-only", task.row_index, case)
    try:
        imgs={k: load_nifti(r[k]) for k in ["fixed_img_path","moving_img_path","warped_img_path"] if pd.notna(r.get(k))}
        segs={k: load_nifti(r[k]) for k in ["fixed_seg_path","moving_seg_path","warped_seg_path"] if pd.notna(r.get(k))}
        get_axis_code_mapping(imgs["fixed_img_path"]); spacing=get_spacing_from_affine(imgs["fixed_img_path"])
        data={k: np.asanyarray(v.dataobj) for k,v in imgs.items()}; sdata={k: np.asanyarray(v.dataobj) for k,v in segs.items()}
        names={"fixed_img_path":"fixed", "moving_img_path":"moving", "warped_img_path":"warped"}
        snames={"fixed_seg_path":"fixed_seg", "moving_seg_path":"moving_seg", "warped_seg_path":"warped_seg"}
        for k, arr in data.items():
            LOGGER.info("[LOAD 3D] %s shape=%s", names[k], arr.shape)
            if arr.ndim != 3:
                reason=_non3d_skip(case, task.row_index, r.get(k), arr); row=base|{"status":"skipped","skip_reason":reason,"runtime_seconds":time.time()-t0}; return {"success":False,"row_index":task.row_index,"case_id":case,"result_rows":[row],"error_rows":[row],"runtime_seconds":time.time()-t0}
        for k, arr in sdata.items():
            LOGGER.info("[LOAD 3D] %s shape=%s", snames[k], arr.shape)
            if arr.ndim != 3:
                reason=_non3d_skip(case, task.row_index, r.get(k), arr); row=base|{"status":"skipped","skip_reason":reason,"runtime_seconds":time.time()-t0}; return {"success":False,"row_index":task.row_index,"case_id":case,"result_rows":[row],"error_rows":[row],"runtime_seconds":time.time()-t0}
        fixed=normalize_intensity_0_1(data["fixed_img_path"], "fixed", case, task.row_index)
        moving=normalize_intensity_0_1(data["moving_img_path"], "moving", case, task.row_index)
        warped=normalize_intensity_0_1(data["warped_img_path"], "warped", case, task.row_index)
        fseg=sdata["fixed_seg_path"]; mseg=sdata["moving_seg_path"]; wseg=sdata["warped_seg_path"]
        row=base.copy()
        if fixed.shape != moving.shape or fixed.shape != warped.shape:
            row.update(status="skipped", skip_reason="shape_mismatch")
        else:
            if task.enable_global: row.update(compute_global_metrics(fixed,moving,warped,case,frame,bins=task.nmi_bins,row_index=task.row_index,device=device))
            if task.enable_seg: row.update(compute_segmentation_metrics(fseg,mseg,wseg,task.labels,spacing,case,frame,row_index=task.row_index,device=device,seg_metric_organs=task.selected_seg_organs,seg_mean_organs=task.selected_seg_mean_organs,verbose_seg_mean=task.verbose_seg_mean,min_mask_volume_voxels=task.min_mask_volume_voxels))
            if task.enable_motion: row.update(compute_organ_ncc_moves(fixed,moving,warped,fseg,mseg,wseg,task.labels,imgs["fixed_img_path"].affine,case,frame,row_index=task.row_index,device=device,ncc_batch_size=task.ncc_batch_size,min_mask_volume_voxels=task.min_mask_volume_voxels,severe_volume_ratio_threshold=task.severe_volume_ratio_threshold))
            if task.enable_vertebra: row.update(compute_vertebra_ncc(fixed,moving,warped,fseg,mseg,wseg,case,frame,row_index=task.row_index,device=device))
        if task.enable_dvf and pd.notna(r.get("transform_path")):
            row.update(compute_dvf_metrics(r["transform_path"], case, row_index=task.row_index, device=device))
        row["runtime_seconds"]=time.time()-t0
        return {"success":row.get("status") == "ok","row_index":task.row_index,"case_id":case,"result_rows":[row],"error_rows":[] if row.get("status") == "ok" else [row],"runtime_seconds":time.time()-t0}
    except (OSError, ValueError, KeyError, RuntimeError, ImportError) as e:
        msg=f"{type(e).__name__}: {e}"; LOGGER.error("[ERROR] case=%s error=%s traceback=%s", case, msg, traceback.format_exc(limit=3)); er=base|{"status":"error","error_message":msg,"runtime_seconds":time.time()-t0}; return {"success":False,"row_index":task.row_index,"case_id":case,"result_rows":[er],"error_rows":[er],"runtime_seconds":time.time()-t0}


def _build_tasks(config: dict, labels: dict[int, str], selected_seg_organs: list[str], selected_seg_mean_organs: list[str], enable_global: bool, enable_seg: bool, enable_dvf: bool, enable_motion: bool, enable_vertebra: bool, use_gpu: bool, requested_device: str, ncc_batch_size: int, verbose_seg_mean: bool, min_mask_volume_voxels: int, severe_volume_ratio_threshold: float, nmi_bins: int) -> list[CaseTask]:
    tasks=[]
    for method, groups in config.items():
        if method in {"label_map", "seg_metric_organs", "seg_mean_organs", "min_mask_volume_voxels", "severe_volume_ratio_threshold", "nmi_bins"}: continue
        for analysis_group, g in groups.items():
            df=normalize_columns(pd.read_csv(g["csv_path"])); total_rows=len(df)
            for row_index, r in df.iterrows():
                tasks.append(CaseTask(method,g.get("center"),g.get("modality"),g.get("task"),g.get("organ"),analysis_group,g.get("csv_path"),int(row_index),total_rows,r.to_dict(),labels,selected_seg_organs,selected_seg_mean_organs,enable_global,enable_seg,enable_dvf,enable_motion,enable_vertebra,use_gpu,requested_device,ncc_batch_size,verbose_seg_mean,min_mask_volume_voxels,severe_volume_ratio_threshold,nmi_bins))
    return tasks


def _finalize_outputs(tasks: list[CaseTask], all_rows: list[dict], errors: list[dict], outdir: Path, progress_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    combined=pd.DataFrame(all_rows)
    for c in REQUIRED_OUTPUT_COLUMNS:
        if c not in combined: combined[c]=np.nan
    combined.to_csv(outdir/"combined_metrics.csv", index=False); LOGGER.info("[SAVE] combined_metrics.csv=%s", outdir/"combined_metrics.csv")
    pd.DataFrame(errors).to_csv(outdir/"error_log.csv", index=False); LOGGER.info("[SAVE] error_log.csv=%s", outdir/"error_log.csv")
    saved_groups=set()
    for task in tasks:
        key=(task.method, task.analysis_group)
        if key in saved_groups:
            continue
        saved_groups.add(key)
        group_rows=[r for r in all_rows if r.get("Method") == task.method and r.get("AnalysisGroup") == task.analysis_group]
        if group_rows:
            p=outdir/f"metrics_{task.method}_{task.center}_{task.modality}_{task.organ}.csv".replace('/','_').replace(' ','_')
            pd.DataFrame(group_rows).to_csv(p,index=False); LOGGER.info("[SAVE] group detailed CSV=%s", p)
    save_summary_from_progress(progress_path, outdir)
    return combined, pd.DataFrame(), pd.DataFrame()


def _process_pool_context(use_gpu: bool, workers: int):
    """Return a multiprocessing context for process pools without initializing CUDA."""
    if use_gpu and workers > 1:
        LOGGER.info("[MP GPU] use_gpu=True num_workers=%s start_method=spawn", workers)
        LOGGER.warning("[MP GPU WARNING] Multiple GPU workers may increase GPU memory usage. If OOM occurs, use --num-workers 1.")
        return mp.get_context("spawn")
    return None


def _run_single_process(tasks: list[CaseTask], progress_path: Path, error_path: Path) -> tuple[list[dict], list[dict], int, int]:
    all_rows=[]; errors=[]; completed=0; failed=0
    for task in tasks:
        result=compute_single_case_task(task); completed += 1; failed += 0 if result.get("success") else 1
        all_rows.extend(result.get("result_rows", [])); errors.extend(result.get("error_rows", [])); append_rows_to_csv(result.get("result_rows", []), progress_path); append_rows_to_csv(result.get("error_rows", []), error_path)
        LOGGER.info("[MP DONE] row=%s case_id=%s success=%s appended_rows=%s runtime_seconds=%s", result.get("row_index"), result.get("case_id"), result.get("success"), len(result.get("result_rows", [])), result.get("runtime_seconds"))
    return all_rows, errors, completed, failed


def compute_from_config(config: dict, output_dir: str|Path, enable_global=True, enable_seg=True, enable_dvf=True, enable_motion=True, enable_vertebra=True, use_gpu: bool = False, requested_device: str = "cuda:0", gpu_metrics: str = "all", ncc_batch_size: int = 64, seg_metric_organs: str | list[str] | None = None, seg_mean_organs: str | list[str] | None = None, verbose_seg_mean: bool = False, min_mask_volume_voxels: int | None = None, severe_volume_ratio_threshold: float | None = None, num_workers: int = 1) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    """Compute metrics for configured CSV rows as pre-split 3D cases with optional multiprocessing."""
    outdir=Path(output_dir); outdir.mkdir(parents=True, exist_ok=True); labels=merged_label_map(config); selected_seg_organs=resolve_seg_metric_organs(config, seg_metric_organs); selected_seg_mean_organs=resolve_seg_mean_organs(config, seg_mean_organs); min_mask_volume_voxels, severe_volume_ratio_threshold = resolve_mask_quality_thresholds(config, min_mask_volume_voxels, severe_volume_ratio_threshold); progress_path=outdir/"detailed_progress.csv"; error_path=outdir/"error_log.csv"; nmi_bins=int(config.get("nmi_bins", 64))
    LOGGER.info("[CONFIG] methods=%s output_dir=%s", list(config.keys()), outdir)
    LOGGER.info("[SEG METRIC] selected individual organ metrics organs=%s", selected_seg_organs)
    LOGGER.info("[SEG METRIC] mean organ set organs=%s", selected_seg_mean_organs)
    LOGGER.info("[MOTION MASK CHECK] min_mask_volume_voxels=%s severe_volume_ratio_threshold=%s", min_mask_volume_voxels, severe_volume_ratio_threshold)
    for method, groups in config.items():
        if method in {"label_map", "seg_metric_organs", "seg_mean_organs", "min_mask_volume_voxels", "severe_volume_ratio_threshold", "nmi_bins"}: continue
        for analysis_group, g in groups.items():
            LOGGER.info("[GROUP START] input_csv=%s method=%s group=%s center=%s modality=%s task=%s organ=%s use_gpu=%s requested_device=%s", g.get("csv_path"), method, analysis_group, g.get('center'), g.get('modality'), g.get('task'), g.get('organ'), use_gpu, requested_device)
    tasks=_build_tasks(config,labels,selected_seg_organs,selected_seg_mean_organs,enable_global,enable_seg,enable_dvf,enable_motion,enable_vertebra,use_gpu,requested_device,ncc_batch_size,verbose_seg_mean,min_mask_volume_voxels,severe_volume_ratio_threshold,nmi_bins)
    all_rows=[]; errors=[]; completed=0; failed=0; workers=max(int(num_workers or 1), 1)
    LOGGER.info("[MP START] num_workers=%s", workers)
    if workers <= 1:
        all_rows, errors, completed, failed = _run_single_process(tasks, progress_path, error_path)
    else:
        try:
            ctx = _process_pool_context(use_gpu, workers)
            executor_kwargs={"max_workers": workers}
            if ctx is not None:
                executor_kwargs["mp_context"] = ctx
            with ProcessPoolExecutor(**executor_kwargs) as ex:
                futs=[]
                for task in tasks:
                    case=str(task.row_dict.get("case_id", task.row_dict.get("run_id", task.row_index))); LOGGER.info("[MP SUBMIT] row=%s case_id=%s", task.row_index, case); futs.append(ex.submit(compute_single_case_task, task))
                for fut in as_completed(futs):
                    try:
                        result=fut.result()
                    except Exception as e:
                        failed += 1; LOGGER.error("[MP ERROR] row=unknown case_id=unknown error=%s", e); result={"success":False,"row_index":None,"case_id":"unknown","result_rows":[],"error_rows":[{"case_id":"unknown","status":"error","error_message":f"{type(e).__name__}: {e}"}],"runtime_seconds":float("nan")}
                    completed += 1; failed += 0 if result.get("success") else 1
                    all_rows.extend(result.get("result_rows", [])); errors.extend(result.get("error_rows", [])); append_rows_to_csv(result.get("result_rows", []), progress_path); append_rows_to_csv(result.get("error_rows", []), error_path)
                    LOGGER.info("[MP DONE] row=%s case_id=%s runtime_seconds=%s", result.get("row_index"), result.get("case_id"), result.get("runtime_seconds"))
        except Exception as e:
            LOGGER.error("[MP ERROR] failed to start or run process pool with num_workers=%s use_gpu=%s error=%s; falling back to num_workers=1", workers, use_gpu, e)
            all_rows, errors, completed, failed = _run_single_process(tasks, progress_path, error_path)
    LOGGER.info("[MP SUMMARY] submitted=%s completed=%s failed=%s", len(tasks), completed, failed)
    return _finalize_outputs(tasks, all_rows, errors, outdir, progress_path)
