"""Publication-style violin plots for registration metrics."""
from __future__ import annotations
import logging
from pathlib import Path
from collections.abc import Iterable
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
LOGGER=logging.getLogger("registration_metrics")
DEFAULT_METRICS=["nmi_warped_fixed","ssim_warped_fixed","lcc_warped_fixed","dice_foreground_warped_fixed","iou_foreground_warped_fixed","hd95_foreground_warped_fixed","assd_foreground_warped_fixed","folding_ratio","jacobian_mean","VertebraNCC_warped_fixed","MovementError","MovementError_AP","MovementError_RL","MovementError_SI","MotionPCC_AllDirections","MotionAMD_AllDirections","MotionMAPE_percent_AllDirections","MotionRMSE_AllDirections","AmplitudeAMD"]
_METADATA_ALIASES={"Method":"method","Center":"center","Modality":"modality","Task":"task","Organ":"organ","AnalysisGroup":"analysis_group","CaseID":"case_id","Frame":"frame"}
_STAT_METADATA_COLUMNS={"case_id","CaseID","Frame","frame","row_index","Method","method","Center","center","Modality","modality","Task","task","Organ","organ","AnalysisGroup","analysis_group","fixed_img_path","moving_img_path","warped_img_path","fixed_seg_path","moving_seg_path","warped_seg_path","transform_path","status","error_message","skip_reason","completed_at","run_id","runtime_seconds","source_csv","source_table_type"}
_STAT_COLUMNS=["metric","count","missing_count","mean","var","std","median","q25","q75","iqr","min","max"]


def _as_path_list(value) -> list[Path]:
    """Convert None, str, Path, list, or tuple into a flat list of Path values."""
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [Path(value)] if str(value) else []
    if isinstance(value, Iterable):
        paths=[]
        for item in value:
            paths.extend(_as_path_list(item))
        return paths
    return [Path(value)]


def load_and_merge_metric_csvs(paths, table_name, logger=None) -> pd.DataFrame:
    """Read CSV files, add source diagnostics, and concatenate with all columns preserved."""
    log=logger or LOGGER; path_list=_as_path_list(paths); frames=[]
    if not path_list:
        raise ValueError(f"[PLOT MERGE] table={table_name} no input CSV files provided")
    for path in path_list:
        if not path.exists():
            log.error("[PLOT LOAD] table=%s path=%s reason=file_not_found", table_name, path)
            raise FileNotFoundError(f"[PLOT LOAD] table={table_name} CSV not found: {path}")
        df=pd.read_csv(path).copy(); df["source_csv"]=str(path); df["source_table_type"]=table_name
        log.info("[PLOT LOAD] table=%s path=%s rows=%s cols=%s", table_name, path, len(df), len(df.columns))
        frames.append(df)
    merged=pd.concat(frames, ignore_index=True, sort=False)
    if merged.empty:
        raise ValueError(f"[PLOT MERGE] table={table_name} all input CSV files are empty")
    log.info("[PLOT MERGE] table=%s n_files=%s total_rows=%s total_cols=%s", table_name, len(path_list), len(merged), len(merged.columns))
    return merged


def standardize_plot_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add lowercase metadata aliases without deleting or overwriting original columns."""
    out=df.copy()
    for src, dst in _METADATA_ALIASES.items():
        if src not in out.columns:
            continue
        if dst not in out.columns:
            out[dst]=out[src]
            LOGGER.info("[STANDARDIZE] added alias column %s from %s", dst, src)
        else:
            missing=out[dst].isna()
            n_missing=int(missing.sum())
            if n_missing:
                out.loc[missing, dst]=out.loc[missing, src]
                LOGGER.info("[STANDARDIZE] existing column %s preserved; filled missing values from %s", dst, src)
    return out


def infer_metric_columns_for_statistics(df: pd.DataFrame) -> list[str]:
    """Return numeric metric columns excluding uppercase and lowercase metadata columns."""
    return [c for c in df.select_dtypes(include="number").columns if c not in _STAT_METADATA_COLUMNS]


def _parse_group_cols(value, df: pd.DataFrame) -> list[str]:
    if value:
        return [c.strip() for c in str(value).split(",") if c.strip() and c.strip() in df.columns]
    upper=["Method","Center","Modality","Task","Organ","AnalysisGroup"]
    lower=["method","center","modality","task","organ","analysis_group"]
    chosen=[c for c in upper if c in df.columns]
    if chosen:
        chosen.extend([c for c in lower if c in df.columns and not any(_METADATA_ALIASES.get(u) == c for u in chosen)])
        return chosen
    return [c for c in lower if c in df.columns]


def _summarize_series(s: pd.Series, total_rows: int) -> dict:
    numeric=pd.to_numeric(s, errors="coerce"); count=int(numeric.count()); q25=numeric.quantile(0.25); q75=numeric.quantile(0.75)
    return {"count":count,"missing_count":int(total_rows-count),"mean":numeric.mean(),"var":numeric.var(),"std":numeric.std(),"median":numeric.median(),"q25":q25,"q75":q75,"iqr":q75-q25 if pd.notna(q25) and pd.notna(q75) else np.nan,"min":numeric.min(),"max":numeric.max()}


def _overall_statistics(df: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    rows=[]
    for metric in metric_cols:
        rows.append({"metric":metric, **_summarize_series(df[metric], len(df))})
    return pd.DataFrame(rows, columns=_STAT_COLUMNS)


def _group_statistics(df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    if not group_cols:
        return pd.DataFrame(columns=["metric", *_STAT_COLUMNS[1:]])
    rows=[]
    for keys, group in df.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys=(keys,)
        base=dict(zip(group_cols, keys))
        for metric in metric_cols:
            rows.append({**base, "metric":metric, **_summarize_series(group[metric], len(group))})
    return pd.DataFrame(rows, columns=[*group_cols, *_STAT_COLUMNS])


def save_plot_statistics(df: pd.DataFrame, output_dir: Path, x: str, hue: str, statistics_group_cols=None) -> None:
    """Save long-format descriptive statistics CSVs for plot input."""
    metric_cols=infer_metric_columns_for_statistics(df); group_cols=_parse_group_cols(statistics_group_cols, df)
    LOGGER.info("[STATS] n_metric_columns=%s", len(metric_cols)); LOGGER.info("[STATS] group_cols=%s", group_cols)
    overall=_overall_statistics(df, metric_cols); group=_group_statistics(df, group_cols, metric_cols)
    x_hue_cols=[c for c in [x, hue] if c in df.columns]
    x_hue=_group_statistics(df, x_hue_cols, metric_cols) if len(x_hue_cols) == len([x, hue]) else pd.DataFrame(columns=[*x_hue_cols, *_STAT_COLUMNS])
    overall_path=output_dir/"plot_statistics_overall.csv"; group_path=output_dir/"plot_statistics_by_group.csv"; x_hue_path=output_dir/"plot_statistics_by_x_hue.csv"
    overall.to_csv(overall_path, index=False); group.to_csv(group_path, index=False); x_hue.to_csv(x_hue_path, index=False)
    LOGGER.info("[STATS] saved overall statistics to %s", overall_path); LOGGER.info("[STATS] saved group statistics to %s", group_path); LOGGER.info("[STATS] saved x/hue statistics to %s", x_hue_path)


def plot_violin(metrics_csv, case_motion_csv=None, output_dir=None, hue: str="center", x: str="modality", metrics: list[str]|None=None, save_statistics: bool=False, statistics_group_cols=None, save_merged_plot_input: bool=True) -> None:
    """Read metric CSV files and save one PNG/PDF/SVG violin plot per metric at 600 DPI."""
    out=Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    metrics_df=standardize_plot_metadata_columns(load_and_merge_metric_csvs(metrics_csv, "metrics", LOGGER))
    case_paths=_as_path_list(case_motion_csv); frames=[metrics_df]; case_motion_df=None
    if case_paths:
        case_motion_df=standardize_plot_metadata_columns(load_and_merge_metric_csvs(case_paths, "case_motion", LOGGER)); frames.append(case_motion_df)
    plot_df=standardize_plot_metadata_columns(pd.concat(frames, ignore_index=True, sort=False))
    LOGGER.info("[PLOT] merged input rows=%s cols=%s", len(plot_df), len(plot_df.columns))
    if save_merged_plot_input:
        metrics_path=out/"merged_metrics_for_plot.csv"; metrics_df.to_csv(metrics_path, index=False); LOGGER.info("[SAVE MERGED] metrics path=%s rows=%s", metrics_path, len(metrics_df))
        if case_motion_df is not None:
            case_path=out/"merged_case_motion_for_plot.csv"; case_motion_df.to_csv(case_path, index=False); LOGGER.info("[SAVE MERGED] case_motion path=%s rows=%s", case_path, len(case_motion_df))
        all_path=out/"merged_all_for_plot.csv"; plot_df.to_csv(all_path, index=False); LOGGER.info("[SAVE MERGED] all path=%s rows=%s", all_path, len(plot_df))
    if save_statistics:
        save_plot_statistics(plot_df, out, x, hue, statistics_group_cols)
    mets=metrics or DEFAULT_METRICS; LOGGER.info("[PLOT] metrics to plot=%s", mets); LOGGER.info("[PLOT] hue=%s, x=%s", hue, x)
    plt.rcParams["font.family"]="Times New Roman"; sns.set_theme(style="whitegrid")
    for m in mets:
        if m not in plot_df.columns:
            LOGGER.info("[PLOT SKIP] metric=%s reason=metric_column_missing", m); continue
        if x not in plot_df.columns or hue not in plot_df.columns:
            LOGGER.info("[PLOT SKIP] metric=%s reason=x_or_hue_column_missing x=%s hue=%s", m, x, hue); continue
        sub=plot_df[[x,hue,m]].dropna(); LOGGER.info("[PLOT] metric=%s x=%s hue=%s valid_rows=%s", m, x, hue, len(sub))
        if sub.empty or pd.to_numeric(sub[m], errors="coerce").dropna().empty:
            LOGGER.info("[PLOT SKIP] metric=%s reason=no_finite_values", m); continue
        fig, ax=plt.subplots(figsize=(6,4)); sns.violinplot(data=sub, x=x, y=m, hue=hue, inner="box", cut=0, ax=ax); ax.set_title(m); fig.tight_layout()
        for ext in ["png","pdf","svg"]:
            path=out/f"{m}.{ext}"; fig.savefig(path, dpi=600); LOGGER.info("[PLOT] save path=%s", path)
        plt.close(fig)
