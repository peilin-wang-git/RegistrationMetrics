"""Publication-style violin plots for registration metrics."""
from __future__ import annotations
import logging
from pathlib import Path
from collections.abc import Iterable
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
LOGGER=logging.getLogger("registration_metrics")
DEFAULT_METRICS=["nmi_warped_fixed","ssim_warped_fixed","lcc_warped_fixed","dice_foreground_warped_fixed","iou_foreground_warped_fixed","hd95_foreground_warped_fixed","assd_foreground_warped_fixed","folding_ratio","jacobian_mean","VertebraNCC_warped_fixed","MovementError","MovementError_AP","MovementError_RL","MovementError_SI","MotionPCC_AllDirections","MotionAMD_AllDirections","MotionMAPE_percent_AllDirections","MotionRMSE_AllDirections","AmplitudeAMD"]
_METADATA_ALIASES={"Method":"method","Center":"center","Modality":"modality","Task":"task","Organ":"organ","AnalysisGroup":"analysis_group","CaseID":"case_id","Frame":"frame"}
_FEATURE_ALIASES={"Method":"method","method":"method","Center":"center","center":"center","Modality":"modality","modality":"modality","Task":"task","task":"task","Organ":"organ","organ":"organ","AnalysisGroup":"analysis_group","analysis_group":"analysis_group"}
_FEATURE_COLUMNS={"method":["method","Method"],"center":["center","Center"],"modality":["modality","Modality"],"task":["task","Task"],"organ":["organ","Organ"],"analysis_group":["analysis_group","AnalysisGroup"]}
_STAT_METADATA_COLUMNS={"case_id","CaseID","Frame","frame","row_index","Method","method","Center","center","Modality","modality","Task","task","Organ","organ","AnalysisGroup","analysis_group","fixed_img_path","moving_img_path","warped_img_path","fixed_seg_path","moving_seg_path","warped_seg_path","transform_path","status","error_message","skip_reason","completed_at","run_id","runtime_seconds","source_csv","source_table_type","_x_group","_shade_group","_color_group"}
_STAT_COLUMNS=["metric","count","missing_count","mean","var","std","median","q25","q75","iqr","min","max"]
_X_COMPOSITE_PRIORITY=["center","organ","modality","task"]
_X_FALLBACK_PRIORITY=["modality","center","task","organ","method"]
_HUE_PRIORITY=["method","center","modality","task","organ"]
_SHADE_PRIORITY=["modality","task","organ","center","method","analysis_group"]


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


def parse_column_list(value: str | None) -> list[str]:
    """Parse comma-separated column names, treating none/None/empty as no columns."""
    if value is None:
        return []
    text=str(value).strip()
    if not text or text.lower() == "none":
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


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


def canonical_feature_name(col: str) -> str:
    """Map metadata aliases such as Method/method to canonical feature names."""
    return _FEATURE_ALIASES.get(col, col)


def _preferred_existing_column(df: pd.DataFrame, feature: str) -> str | None:
    for col in _FEATURE_COLUMNS.get(feature, [feature]):
        if col in df.columns:
            return col
    return None


def _feature_has_variation(df: pd.DataFrame, feature: str) -> bool:
    col=_preferred_existing_column(df, feature)
    return bool(col and df[col].dropna().nunique() >= 2)


def _validate_columns_exist(df: pd.DataFrame, cols: list[str], option_name: str) -> None:
    missing=[col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"{option_name} columns not found: {missing}. Available columns: {list(df.columns)}")


def build_composite_group_column(df: pd.DataFrame, cols: list[str], output_col: str, sep: str = " | ", unknown_value: str = "Unknown") -> pd.DataFrame:
    """Build an internal grouping column by concatenating multiple metadata columns."""
    out=df.copy()
    if not cols:
        return out
    pieces=[out[col].where(out[col].notna(), unknown_value).astype(str) for col in cols]
    out[output_col]=pieces[0] if len(pieces) == 1 else pieces[0].str.cat(pieces[1:], sep=sep)
    return out


def infer_x_columns(df: pd.DataFrame, user_x: str | None = None, auto_plot_mapping: bool = True) -> list[str]:
    """Infer x-axis metadata columns, supporting comma-separated composite x groups."""
    parsed=parse_column_list(user_x)
    LOGGER.info("[PLOT X] user_x=%s parsed_x_cols=%s", user_x, parsed)
    if parsed:
        _validate_columns_exist(df, parsed, "x")
        return parsed
    if not auto_plot_mapping:
        raise ValueError("No --x columns specified and auto plot mapping is disabled")
    analysis_col=_preferred_existing_column(df, "analysis_group")
    if analysis_col and df[analysis_col].dropna().nunique() >= 2:
        return [analysis_col]
    composite=[_preferred_existing_column(df, f) for f in _X_COMPOSITE_PRIORITY]
    composite=[c for c in composite if c]
    if composite and any(df[c].dropna().nunique() >= 2 for c in composite):
        return composite
    for feature in _X_FALLBACK_PRIORITY:
        col=_preferred_existing_column(df, feature)
        if col and df[col].dropna().nunique() >= 2:
            return [col]
    for feature in _X_FALLBACK_PRIORITY:
        col=_preferred_existing_column(df, feature)
        if col:
            return [col]
    raise ValueError("Could not infer plot x column; please specify --x with one or more metadata columns.")


def get_consumed_metadata_features(x_cols: list[str], hue: str | None) -> set[str]:
    """Return canonical metadata features consumed by x columns and hue."""
    consumed={canonical_feature_name(col) for col in x_cols if col}
    if hue:
        consumed.add(canonical_feature_name(hue))
    if "analysis_group" in consumed:
        consumed.update({"analysis_group","center","organ","task"})
    return consumed


def infer_hue_column(df: pd.DataFrame, x_cols: list[str], user_hue: str | None = None) -> str | None:
    """Infer a single hue metadata column that is not consumed by the x columns."""
    parsed=parse_column_list(user_hue)
    if parsed:
        if len(parsed) > 1:
            raise ValueError("--hue supports a single column; use one metadata column or --hue none.")
        _validate_columns_exist(df, parsed, "hue")
        LOGGER.info("[PLOT HUE] user_hue=%s inferred_hue=%s", user_hue, parsed[0])
        return parsed[0]
    if user_hue is not None and not parsed:
        LOGGER.info("[PLOT HUE] user_hue=%s inferred_hue=None", user_hue)
        return None
    consumed=get_consumed_metadata_features(x_cols, None)
    for feature in _HUE_PRIORITY:
        if feature in consumed:
            LOGGER.info("[PLOT HUE] skipped %s because it is already used in x_cols", feature)
            continue
        col=_preferred_existing_column(df, feature)
        if col and df[col].dropna().nunique() >= 2:
            LOGGER.info("[PLOT HUE] user_hue=%s inferred_hue=%s", user_hue, col)
            return col
    LOGGER.info("[PLOT HUE] final_hue=None reason=no unused metadata column with >=2 unique values")
    return None


def infer_plot_mapping(df: pd.DataFrame, x: str | None = None, hue: str | None = None) -> tuple[list[str], str, str | None]:
    """Infer x columns, final x plotting column, and hue column."""
    x_cols=infer_x_columns(df, x)
    final_hue=infer_hue_column(df, x_cols, hue)
    if len(x_cols) > 1:
        LOGGER.info("[PLOT X] composite x enabled final_x=_x_group")
        return x_cols, "_x_group", final_hue
    return x_cols, x_cols[0], final_hue


def _varying_metadata_features(df: pd.DataFrame) -> set[str]:
    return {feature for feature in _FEATURE_COLUMNS if _feature_has_variation(df, feature)}


def infer_shade_by(df: pd.DataFrame, x_cols=None, hue: str | None = None, shade_by: str | None = "auto", max_shade_levels: int = 8, x=None) -> list[str]:
    """Infer secondary grouping columns from metadata not consumed by x/hue."""
    if x_cols is None and x is not None:
        x_cols=x
    if isinstance(x_cols, str) or x_cols is None:
        x_cols=[] if x_cols is None else [x_cols]
    LOGGER.info("[PLOT SHADE] user_shade_by=%s", shade_by)
    if shade_by is None or str(shade_by).strip() == "" or str(shade_by).strip().lower() == "none":
        LOGGER.info("[PLOT SHADE] user_shade_by=none; shade encoding disabled")
        return []
    consumed=get_consumed_metadata_features(list(x_cols), hue)
    varying=_varying_metadata_features(df)
    remaining=varying-consumed
    LOGGER.info("[PLOT SHADE] varying_metadata_features=%s", varying)
    LOGGER.info("[PLOT SHADE] consumed_features=%s", consumed)
    LOGGER.info("[PLOT SHADE] remaining_features=%s", remaining)
    if str(shade_by).strip().lower() != "auto":
        cols=parse_column_list(shade_by)
        _validate_columns_exist(df, cols, "shade_by")
        for col in cols:
            feature=canonical_feature_name(col)
            if feature in consumed:
                raise ValueError(f"shade_by column {col} is already used by x or hue; please choose another column or use --shade-by none.")
        return cols
    if not remaining:
        LOGGER.info("[PLOT SHADE] shade disabled because x and hue already consume all varying metadata features")
        return []
    selected=[]
    for feature in _SHADE_PRIORITY:
        if feature not in remaining:
            continue
        col=_preferred_existing_column(df, feature)
        if col and df[col].dropna().nunique() >= 2:
            selected.append(col)
        if len(selected) >= 2:
            break
    if not selected:
        LOGGER.info("[PLOT SHADE] no valid shade_by column found; using base hue colors only")
    LOGGER.info("[PLOT SHADE] inferred_shade_by=%s", selected)
    if selected:
        tmp=build_shade_group_column(df, selected)
        levels=list(pd.Series(tmp["_shade_group"]).dropna().astype(str).unique())
        if len(levels) > max_shade_levels:
            LOGGER.warning("[PLOT SHADE WARNING] shade_group has %s levels > max_shade_levels=%s; colors may be hard to distinguish", len(levels), max_shade_levels)
    return selected


def build_shade_group_column(df: pd.DataFrame, shade_by_cols: list[str], output_col: str = "_shade_group") -> pd.DataFrame:
    """Build an internal shade grouping column from one or more metadata columns."""
    out=build_composite_group_column(df, shade_by_cols, output_col)
    if not shade_by_cols:
        return out
    levels=list(pd.Series(out[output_col]).dropna().astype(str).unique())
    LOGGER.info("[PLOT SHADE] shade_by_cols=%s", shade_by_cols)
    LOGGER.info("[PLOT SHADE] shade_group_levels=%s", levels)
    LOGGER.info("[PLOT SHADE] n_shade_levels=%s levels=%s", len(levels), levels)
    return out


def adjust_lightness(color, factor):
    """Adjust color lightness by blending with black for factor<1 or white for factor>1."""
    rgb=np.array(to_rgb(color), dtype=float)
    if factor < 1:
        return tuple(np.clip(rgb * max(factor, 0.35), 0, 1))
    amount=min(factor - 1, 0.55)
    return tuple(np.clip(rgb + (1 - rgb) * amount, 0, 0.92))


def _shade_factors(n: int) -> list[float]:
    if n <= 1:
        return [1.0]
    if n == 2:
        return [0.75, 1.20]
    if n == 3:
        return [0.70, 1.00, 1.30]
    return list(np.linspace(0.65, 1.35, n))


def make_hue_shade_palette(df: pd.DataFrame, hue: str | None, shade_group_col: str | None = "_shade_group", base_palette: str = "tab10") -> dict:
    """Return palette mapping hue or combined hue+shade values to colors."""
    out=df
    has_shade=shade_group_col is not None and shade_group_col in out.columns
    if hue is not None and hue in out.columns and has_shade:
        out["_color_group"]=out[hue].where(out[hue].notna(), "Unknown").astype(str)+" | "+out[shade_group_col].where(out[shade_group_col].notna(), "Unknown").astype(str)
        hue_levels=list(out[hue].where(out[hue].notna(), "Unknown").astype(str).unique())
        base_colors=sns.color_palette(base_palette, n_colors=max(len(hue_levels), 1))
        palette={}
        for base_color, hue_level in zip(base_colors, hue_levels):
            mask=out[hue].where(out[hue].notna(), "Unknown").astype(str) == hue_level
            shade_levels=list(out.loc[mask, shade_group_col].where(out.loc[mask, shade_group_col].notna(), "Unknown").astype(str).unique())
            for factor, shade_level in zip(_shade_factors(len(shade_levels)), shade_levels):
                palette[f"{hue_level} | {shade_level}"]=adjust_lightness(base_color, factor)
        LOGGER.info("[PLOT COLOR] hue=%s hue_levels=%s", hue, hue_levels)
        LOGGER.info("[PLOT COLOR] color_group_col=_color_group n_color_groups=%s", len(palette))
        LOGGER.info("[PLOT COLOR] same hue levels use same color family with different lightness")
        return palette
    if has_shade:
        out["_color_group"]=out[shade_group_col].where(out[shade_group_col].notna(), "Unknown").astype(str)
        levels=list(out["_color_group"].unique()); base_color=sns.color_palette("crest", n_colors=max(len(levels), 1))[min(2, max(len(levels)-1, 0))]
        palette={level:adjust_lightness(base_color, factor) for factor, level in zip(_shade_factors(len(levels)), levels)}
        LOGGER.info("[PLOT COLOR] hue=%s hue_levels=%s", hue, [])
        LOGGER.info("[PLOT COLOR] color_group_col=_color_group n_color_groups=%s", len(palette))
        return palette
    if hue is not None and hue in out.columns:
        levels=list(out[hue].where(out[hue].notna(), "Unknown").astype(str).unique())
        out[hue]=out[hue].where(out[hue].notna(), "Unknown").astype(str)
        LOGGER.info("[PLOT COLOR] hue=%s hue_levels=%s", hue, levels)
        return dict(zip(levels, sns.color_palette(base_palette, n_colors=max(len(levels), 1))))
    return {}


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


def _dedupe_cols(cols: list[str | None]) -> list[str]:
    out=[]
    for col in cols:
        if col and col not in out:
            out.append(col)
    return out


def save_plot_statistics(df: pd.DataFrame, output_dir: Path, x: str | None, hue: str | None, statistics_group_cols=None, shade_group_col: str | None = None, x_cols: list[str] | None = None, shade_by_cols: list[str] | None = None) -> None:
    """Save long-format descriptive statistics CSVs for plot input."""
    metric_cols=infer_metric_columns_for_statistics(df); group_cols=_parse_group_cols(statistics_group_cols, df)
    LOGGER.info("[STATS] n_metric_columns=%s", len(metric_cols)); LOGGER.info("[STATS] group_cols=%s", group_cols)
    overall=_overall_statistics(df, metric_cols); group=_group_statistics(df, group_cols, metric_cols)
    x_hue_cols=_dedupe_cols([*(x_cols or []), x, hue])
    x_hue_cols=[c for c in x_hue_cols if c in df.columns]
    x_hue=_group_statistics(df, x_hue_cols, metric_cols) if x_hue_cols else pd.DataFrame(columns=[*_STAT_COLUMNS])
    overall_path=output_dir/"plot_statistics_overall.csv"; group_path=output_dir/"plot_statistics_by_group.csv"; x_hue_path=output_dir/"plot_statistics_by_x_hue.csv"
    overall.to_csv(overall_path, index=False); group.to_csv(group_path, index=False); x_hue.to_csv(x_hue_path, index=False)
    LOGGER.info("[STATS] saved overall statistics to %s", overall_path); LOGGER.info("[STATS] saved group statistics to %s", group_path); LOGGER.info("[STATS] saved x/hue statistics to %s", x_hue_path)
    if shade_group_col and shade_group_col in df.columns:
        shade_cols=_dedupe_cols([*(x_cols or []), x, hue, *(shade_by_cols or []), shade_group_col])
        shade_cols=[c for c in shade_cols if c in df.columns]
        shade_stats=_group_statistics(df, shade_cols, metric_cols)
        shade_path=output_dir/"plot_statistics_by_x_hue_shade.csv"; shade_stats.to_csv(shade_path, index=False)
        LOGGER.info("[STATS] saved x/hue/shade statistics to %s", shade_path)


def plot_violin(metrics_csv, case_motion_csv=None, output_dir=None, hue: str|None=None, x: str|None=None, metrics: list[str]|None=None, save_statistics: bool=False, statistics_group_cols=None, save_merged_plot_input: bool=True, shade_by: str|None="auto", max_shade_levels: int=8) -> None:
    """Read metric CSV files and save one PNG/PDF/SVG violin plot per metric at 600 DPI."""
    out=Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    metrics_df=standardize_plot_metadata_columns(load_and_merge_metric_csvs(metrics_csv, "metrics", LOGGER))
    case_paths=_as_path_list(case_motion_csv); frames=[metrics_df]; case_motion_df=None
    if case_paths:
        case_motion_df=standardize_plot_metadata_columns(load_and_merge_metric_csvs(case_paths, "case_motion", LOGGER)); frames.append(case_motion_df)
    plot_df=standardize_plot_metadata_columns(pd.concat(frames, ignore_index=True, sort=False))
    x_cols, final_x, final_hue=infer_plot_mapping(plot_df, x, hue)
    if final_x == "_x_group":
        plot_df=build_composite_group_column(plot_df, x_cols, "_x_group")
        examples=list(plot_df["_x_group"].dropna().astype(str).unique()[:5])
        LOGGER.info("[PLOT X] composite x enabled final_x=_x_group n_x_groups=%s", plot_df["_x_group"].nunique(dropna=True))
        LOGGER.info("[PLOT X] x_group examples=%s", examples)
    shade_by_cols=infer_shade_by(plot_df, x_cols, final_hue, shade_by, max_shade_levels)
    plot_df=build_shade_group_column(plot_df, shade_by_cols)
    shade_col="_shade_group" if shade_by_cols else None
    palette=make_hue_shade_palette(plot_df, final_hue, shade_col)
    plot_hue="_color_group" if shade_by_cols and "_color_group" in plot_df.columns else final_hue
    LOGGER.info("[PLOT] merged input rows=%s cols=%s", len(plot_df), len(plot_df.columns))
    if save_merged_plot_input:
        metrics_path=out/"merged_metrics_for_plot.csv"; metrics_df.to_csv(metrics_path, index=False); LOGGER.info("[SAVE MERGED] metrics path=%s rows=%s", metrics_path, len(metrics_df))
        if case_motion_df is not None:
            case_path=out/"merged_case_motion_for_plot.csv"; case_motion_df.to_csv(case_path, index=False); LOGGER.info("[SAVE MERGED] case_motion path=%s rows=%s", case_path, len(case_motion_df))
        all_path=out/"merged_all_for_plot.csv"; plot_df.to_csv(all_path, index=False); LOGGER.info("[SAVE MERGED] all path=%s rows=%s", all_path, len(plot_df))
    if save_statistics:
        save_plot_statistics(plot_df, out, final_x, final_hue, statistics_group_cols, shade_col, x_cols, shade_by_cols)
    mets=metrics or DEFAULT_METRICS; LOGGER.info("[PLOT] metrics to plot=%s", mets); LOGGER.info("[PLOT] hue=%s, x=%s", final_hue, final_x)
    plt.rcParams["font.family"]="Times New Roman"; sns.set_theme(style="whitegrid")
    for m in mets:
        if m not in plot_df.columns:
            LOGGER.info("[PLOT SKIP] metric=%s reason=metric_column_missing", m); continue
        if final_x not in plot_df.columns or (plot_hue and plot_hue not in plot_df.columns):
            LOGGER.info("[PLOT SKIP] metric=%s reason=x_or_hue_column_missing x=%s hue=%s", m, final_x, plot_hue); continue
        cols=[final_x,m]+([plot_hue] if plot_hue else [])
        sub=plot_df[cols].dropna(); LOGGER.info("[PLOT] metric=%s x=%s hue=%s valid_rows=%s", m, final_x, plot_hue, len(sub))
        if sub.empty or pd.to_numeric(sub[m], errors="coerce").dropna().empty:
            LOGGER.info("[PLOT SKIP] metric=%s reason=no_finite_values", m); continue
        fig, ax=plt.subplots(figsize=(6,4)); sns.violinplot(data=sub, x=final_x, y=m, hue=plot_hue, palette=palette or None, inner="box", cut=0, ax=ax); ax.set_title(m); fig.tight_layout()
        for ext in ["png","pdf","svg"]:
            path=out/f"{m}.{ext}"; fig.savefig(path, dpi=600); LOGGER.info("[PLOT] save path=%s", path)
        plt.close(fig)
