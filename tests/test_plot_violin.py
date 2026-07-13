import numpy as np
import pandas as pd
from registration_metrics.cli import build_parser
from registration_metrics.plot_violin import plot_violin


def _write_metrics(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def test_plot_accepts_multiple_metrics_csv(tmp_path):
    csv1 = tmp_path / "a.csv"
    csv2 = tmp_path / "b.csv"
    _write_metrics(csv1, [{"Method":"A", "Center":"C1", "Modality":"M1", "nmi_warped_fixed":0.8}])
    _write_metrics(csv2, [{"Method":"B", "Center":"C2", "Modality":"M2", "nmi_warped_fixed":0.9}])
    out = tmp_path / "figures"

    plot_violin(metrics_csv=[csv1, csv2], output_dir=out)

    merged = pd.read_csv(out / "merged_all_for_plot.csv")
    assert len(merged) == 2


def test_plot_cli_accepts_multiple_metrics_csv():
    args = build_parser().parse_args([
        "plot", "--metrics-csv", "a.csv", "b.csv", "c.csv", "--output-dir", "figures"
    ])
    assert args.metrics_csv == ["a.csv", "b.csv", "c.csv"]


def test_case_motion_multiple_csv_concat(tmp_path):
    m1 = tmp_path / "m1.csv"; m2 = tmp_path / "m2.csv"
    c1 = tmp_path / "c1.csv"; c2 = tmp_path / "c2.csv"
    _write_metrics(m1, [{"Method":"A", "Center":"C", "Modality":"M", "nmi_warped_fixed":0.8}])
    _write_metrics(m2, [{"Method":"B", "Center":"C", "Modality":"M", "nmi_warped_fixed":0.9}])
    _write_metrics(c1, [{"Method":"A", "Center":"C", "Modality":"M", "MovementError":1.0}])
    _write_metrics(c2, [{"Method":"B", "Center":"C", "Modality":"M", "MovementError":2.0}])

    out = tmp_path / "figures"
    plot_violin(metrics_csv=[m1, m2], case_motion_csv=[c1, c2], output_dir=out)

    merged = pd.read_csv(out / "merged_all_for_plot.csv")
    assert set(merged["source_table_type"]) == {"metrics", "case_motion"}
    assert len(merged) == 4


def test_plot_statistics_outputs(tmp_path):
    csv = tmp_path / "metrics.csv"
    _write_metrics(csv, [
        {"Method":"A", "Center":"C", "Modality":"M", "nmi_warped_fixed":0.8},
        {"Method":"A", "Center":"C", "Modality":"M", "nmi_warped_fixed":1.0},
    ])
    out = tmp_path / "figures"

    plot_violin(metrics_csv=csv, output_dir=out, save_statistics=True)

    assert (out / "plot_statistics_overall.csv").exists()
    assert (out / "plot_statistics_by_group.csv").exists()
    assert (out / "plot_statistics_by_x_hue.csv").exists()


def test_statistics_excludes_frame(tmp_path):
    csv = tmp_path / "metrics.csv"
    _write_metrics(csv, [{"Method":"A", "Center":"C", "Modality":"M", "Frame":1, "frame":1, "nmi_warped_fixed":0.8}])
    out = tmp_path / "figures"

    plot_violin(metrics_csv=csv, output_dir=out, save_statistics=True)

    stats = pd.read_csv(out / "plot_statistics_overall.csv")
    assert "Frame" not in set(stats["metric"])
    assert "frame" not in set(stats["metric"])


def test_statistics_nan_safe(tmp_path):
    csv = tmp_path / "metrics.csv"
    _write_metrics(csv, [
        {"Method":"A", "Center":"C", "Modality":"M", "nmi_warped_fixed":None},
        {"Method":"A", "Center":"C", "Modality":"M", "nmi_warped_fixed":None},
    ])
    out = tmp_path / "figures"

    plot_violin(metrics_csv=csv, output_dir=out, save_statistics=True)

    stats = pd.read_csv(out / "plot_statistics_overall.csv")
    row = stats.loc[stats["metric"] == "nmi_warped_fixed"].iloc[0]
    assert int(row["count"]) == 0
    assert int(row["missing_count"]) == 2
    assert pd.isna(row["mean"])


def test_missing_metric_column_skipped(tmp_path):
    csv1 = tmp_path / "a.csv"; csv2 = tmp_path / "b.csv"
    _write_metrics(csv1, [{"Method":"A", "Center":"C", "Modality":"M", "nmi_warped_fixed":0.8}])
    _write_metrics(csv2, [{"Method":"B", "Center":"C", "Modality":"M", "ssim_warped_fixed":0.9}])
    out = tmp_path / "figures"

    plot_violin(metrics_csv=[csv1, csv2], output_dir=out, save_statistics=True)

    stats = pd.read_csv(out / "plot_statistics_overall.csv")
    assert {"nmi_warped_fixed", "ssim_warped_fixed"}.issubset(set(stats["metric"]))


def test_single_csv_backward_compatible(tmp_path):
    csv = tmp_path / "metrics.csv"
    _write_metrics(csv, [{"Method":"A", "Center":"C", "Modality":"M", "nmi_warped_fixed":0.8}])
    out = tmp_path / "figures"

    plot_violin(metrics_csv=csv, output_dir=out, save_statistics=True)

    assert (out / "merged_metrics_for_plot.csv").exists()
    assert (out / "merged_all_for_plot.csv").exists()
    assert (out / "nmi_warped_fixed.png").exists()

from registration_metrics.plot_violin import (
    build_shade_group_column,
    infer_shade_by,
    make_hue_shade_palette,
)


def test_infer_shade_by_auto_excludes_x_and_hue():
    df = pd.DataFrame({
        "method": ["DDEM", "FewShot", "DDEM", "FewShot"],
        "center": ["A", "A", "B", "B"],
        "modality": ["T1w", "T2w", "T1w", "T2w"],
    })
    assert infer_shade_by(df, x="method", hue="center", shade_by="auto") == ["modality"]


def test_infer_shade_by_none():
    df = pd.DataFrame({"method": ["A", "B"], "modality": ["T1w", "T2w"]})
    assert infer_shade_by(df, x="method", hue=None, shade_by="none") == []


def test_manual_shade_by_multiple_columns():
    df = pd.DataFrame({"modality": ["T1w"], "task": ["Liver"]})
    assert infer_shade_by(df, x=None, hue=None, shade_by="modality,task") == ["modality"]
    out = build_shade_group_column(df, ["modality", "task"])
    assert out["_shade_group"].iloc[0] == "T1w"


def test_color_group_combines_hue_and_shade():
    df = pd.DataFrame({"center": ["Center A", "Center A"], "modality": ["T1w", "T2w"]})
    df = build_shade_group_column(df, ["modality"])
    make_hue_shade_palette(df, hue="center")
    assert set(df["_color_group"]) == {"Center A | T1w", "Center A | T2w"}


def test_same_hue_same_color_family_different_shades():
    df = pd.DataFrame({
        "center": ["Center A", "Center A", "Center B", "Center B"],
        "modality": ["T1w", "T2w", "T1w", "T2w"],
    })
    df = build_shade_group_column(df, ["modality"])
    palette = make_hue_shade_palette(df, hue="center")
    a_t1 = np.array(palette["Center A | T1w"])
    a_t2 = np.array(palette["Center A | T2w"])
    b_t1 = np.array(palette["Center B | T1w"])
    assert not np.allclose(a_t1, a_t2)
    assert not np.allclose(a_t1, b_t1)
    assert np.linalg.norm(a_t1 - a_t2) < np.linalg.norm(a_t1 - b_t1)


def test_plot_with_shade_by_auto(tmp_path):
    csv1 = tmp_path / "a.csv"
    csv2 = tmp_path / "b.csv"
    _write_metrics(csv1, [{"Method":"DDEM", "Center":"A", "Modality":"T1w", "nmi_warped_fixed":0.8}])
    _write_metrics(csv2, [{"Method":"FewShot", "Center":"B", "Modality":"T2w", "nmi_warped_fixed":0.9}])
    out = tmp_path / "figures"

    plot_violin(metrics_csv=[csv1, csv2], output_dir=out, x="method", hue="center", save_statistics=True)

    assert (out / "merged_all_for_plot.csv").exists()
    assert (out / "plot_statistics_overall.csv").exists()
    assert (out / "nmi_warped_fixed.png").exists()


def test_statistics_excludes_internal_shade_columns(tmp_path):
    csv = tmp_path / "metrics.csv"
    _write_metrics(csv, [{"Method":"DDEM", "Center":"A", "Modality":"T1w", "_shade_group":1, "_color_group":2, "nmi_warped_fixed":0.8}])
    out = tmp_path / "figures"

    plot_violin(metrics_csv=csv, output_dir=out, x="method", hue="center", shade_by="modality", save_statistics=True)

    stats = pd.read_csv(out / "plot_statistics_overall.csv")
    assert "_shade_group" not in set(stats["metric"])
    assert "_color_group" not in set(stats["metric"])


def test_plot_statistics_by_x_hue_shade(tmp_path):
    csv = tmp_path / "metrics.csv"
    _write_metrics(csv, [
        {"Method":"DDEM", "Center":"A", "Modality":"T1w", "nmi_warped_fixed":0.8},
        {"Method":"DDEM", "Center":"A", "Modality":"T2w", "nmi_warped_fixed":0.9},
    ])
    out = tmp_path / "figures"

    plot_violin(metrics_csv=csv, output_dir=out, x="method", hue="center", shade_by="modality", save_statistics=True)

    assert (out / "plot_statistics_by_x_hue_shade.csv").exists()

from registration_metrics.plot_violin import (
    build_composite_group_column,
    get_consumed_metadata_features,
    parse_column_list,
)


def test_parse_x_multiple_columns():
    assert parse_column_list("center,organ,modality") == ["center", "organ", "modality"]
    assert parse_column_list(" Center, Organ , Modality ") == ["Center", "Organ", "Modality"]
    assert parse_column_list("none") == []


def test_build_x_group():
    df = pd.DataFrame({"center": ["A"], "organ": ["Liver"], "modality": ["T1w"]})
    out = build_composite_group_column(df, ["center", "organ", "modality"], "_x_group")
    assert out["_x_group"].iloc[0] == "A | Liver | T1w"


def test_composite_x_consumes_multiple_features():
    consumed = get_consumed_metadata_features(["center", "organ", "modality"], "method")
    assert {"center", "organ", "modality", "method"}.issubset(consumed)


def test_analysis_group_consumes_center_organ_task():
    consumed = get_consumed_metadata_features(["analysis_group"], None)
    assert {"analysis_group", "center", "organ", "modality"}.issubset(consumed)


def test_shade_disabled_when_x_hue_consume_all_varying_features():
    df = pd.DataFrame({
        "method": ["DDEM", "FewShot"],
        "center": ["A", "B"],
        "modality": ["T1w", "T2w"],
    })
    assert infer_shade_by(df, x_cols=["center", "modality"], hue="method", shade_by="auto") == []


def test_shade_auto_uses_remaining_features():
    df = pd.DataFrame({
        "method": ["DDEM", "FewShot", "DDEM", "FewShot"],
        "center": ["A", "A", "B", "B"],
        "modality": ["T1w", "T2w", "T1w", "T2w"],
        "task": ["Liver", "Kidney", "Liver", "Kidney"],
    })
    assert infer_shade_by(df, x_cols=["center"], hue="method", shade_by="auto") == ["modality"]


def test_manual_shade_by_repeated_with_x_raises():
    df = pd.DataFrame({
        "method": ["DDEM", "FewShot"],
        "center": ["A", "B"],
        "modality": ["T1w", "T2w"],
    })
    import pytest
    with pytest.raises(ValueError):
        infer_shade_by(df, x_cols=["center", "modality"], hue="method", shade_by="modality")


def test_plot_composite_x_no_shade(tmp_path):
    csv1 = tmp_path / "a.csv"
    csv2 = tmp_path / "b.csv"
    rows1 = [{"Method":"DDEM", "Center":"A", "Organ":"Liver", "Modality":"T1w", "nmi_warped_fixed":0.8}]
    rows2 = [{"Method":"FewShot", "Center":"B", "Organ":"Kidney", "Modality":"T2w", "nmi_warped_fixed":0.9}]
    _write_metrics(csv1, rows1); _write_metrics(csv2, rows2)
    out = tmp_path / "figures"

    plot_violin(metrics_csv=[csv1, csv2], output_dir=out, x="center,organ,modality", hue="method", save_statistics=True)

    merged = pd.read_csv(out / "merged_all_for_plot.csv")
    assert "_x_group" in merged.columns
    assert "_color_group" not in merged.columns
    assert (out / "nmi_warped_fixed.png").exists()
    assert (out / "plot_statistics_by_x_hue.csv").exists()


def test_statistics_with_composite_x_contains_original_x_cols(tmp_path):
    csv = tmp_path / "metrics.csv"
    _write_metrics(csv, [
        {"Method":"DDEM", "Center":"A", "Organ":"Liver", "Modality":"T1w", "nmi_warped_fixed":0.8},
        {"Method":"FewShot", "Center":"B", "Organ":"Kidney", "Modality":"T2w", "nmi_warped_fixed":0.9},
    ])
    out = tmp_path / "figures"

    plot_violin(metrics_csv=csv, output_dir=out, x="center,organ,modality", hue="method", save_statistics=True)

    stats = pd.read_csv(out / "plot_statistics_by_x_hue.csv")
    assert {"center", "organ", "modality", "_x_group"}.issubset(set(stats.columns))


def test_internal_group_columns_excluded_from_metric_columns(tmp_path):
    csv = tmp_path / "metrics.csv"
    _write_metrics(csv, [{"Method":"DDEM", "Center":"A", "Organ":"Liver", "Modality":"T1w", "_x_group":1, "_shade_group":2, "_color_group":3, "nmi_warped_fixed":0.8}])
    out = tmp_path / "figures"

    plot_violin(metrics_csv=csv, output_dir=out, x="center,organ,modality", hue="method", save_statistics=True)

    stats = pd.read_csv(out / "plot_statistics_overall.csv")
    assert not {"_x_group", "_shade_group", "_color_group"}.intersection(set(stats["metric"]))

from registration_metrics.plot_violin import (
    _apply_plot_layout,
    _figure_size_for_x_groups,
    _save_plot_figure,
    infer_figure_size,
    infer_layout_adjustment,
    parse_aspect_ratio,
)
import matplotlib.pyplot as plt


def test_plot_layout_moves_legend_outside_and_rotates_xticks():
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([0, 1], [0, 1], label="Group A")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Long label A", "Long label B"])
    ax.legend(title="hue")

    _apply_plot_layout(fig, ax, has_legend=True)

    legend = ax.get_legend()
    assert legend is not None
    assert legend._loc == 2  # upper left
    assert tuple(legend.get_bbox_to_anchor()._bbox.p1) == (1.02, 1.0)
    assert all(label.get_rotation() == 12 for label in ax.get_xticklabels())
    assert all(label.get_horizontalalignment() == "right" for label in ax.get_xticklabels())
    plt.close(fig)


def test_save_plot_figure_uses_tight_bbox(monkeypatch, tmp_path):
    fig, _ = plt.subplots()
    captured = {}

    def fake_savefig(path, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(fig, "savefig", fake_savefig)
    _save_plot_figure(fig, tmp_path / "plot.png")

    assert captured["bbox_inches"] == "tight"
    assert captured["pad_inches"] == 0.2
    assert captured["dpi"] == 600
    plt.close(fig)


def test_figure_size_expands_for_many_composite_x_groups():
    small = _figure_size_for_x_groups(3)
    large = _figure_size_for_x_groups(20)
    assert small[0] >= 12
    assert large[0] > small[0]
    assert large[0] <= 32
    assert large[1] >= 6


def test_plot_layout_without_legend_still_rotates_xticks():
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([0, 1], [0, 1])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["A", "B"])

    _apply_plot_layout(fig, ax, has_legend=False)

    assert ax.get_legend() is None
    assert all(label.get_rotation() == 12 for label in ax.get_xticklabels())
    plt.close(fig)


def test_parse_aspect_ratio():
    assert parse_aspect_ratio("16:9") == (16, 9)
    assert parse_aspect_ratio("4:3") == (4, 3)
    assert parse_aspect_ratio(None) is None


def test_manual_figsize_overrides_auto():
    assert infer_figure_size(20, 20, 40, True, fig_width=24, fig_height=10) == (24, 10)


def test_auto_figsize_increases_with_x_groups():
    small = infer_figure_size(4, 0, 8, False)
    large = infer_figure_size(20, 0, 8, False)
    assert large[0] > small[0]


def test_auto_figsize_increases_with_long_labels():
    short = infer_figure_size(8, 0, 10, False)
    long = infer_figure_size(8, 0, 45, False)
    assert long[0] > short[0]
    _, bottom_short = infer_layout_adjustment(10, False, 0, False)
    _, bottom_long = infer_layout_adjustment(45, False, 0, True)
    assert bottom_long > bottom_short


def test_auto_figsize_increases_when_legend_exists():
    no_legend = infer_figure_size(8, 0, 10, False)
    with_legend = infer_figure_size(8, 20, 10, True)
    right_many, _ = infer_layout_adjustment(10, True, 20, False)
    right_few, _ = infer_layout_adjustment(10, True, 4, False)
    assert with_legend[0] > no_legend[0]
    assert right_many < right_few


def test_plot_with_manual_figsize(tmp_path):
    csv = tmp_path / "metrics.csv"
    _write_metrics(csv, [{"Method":"DDEM", "Center":"A", "Modality":"T1w", "nmi_warped_fixed":0.8}])
    out = tmp_path / "figures"

    plot_violin(metrics_csv=csv, output_dir=out, x="modality", hue="method", fig_width=24, fig_height=10)

    assert (out / "nmi_warped_fixed.png").exists()


def test_plot_with_aspect_ratio_only(tmp_path):
    csv = tmp_path / "metrics.csv"
    _write_metrics(csv, [
        {"Method":"DDEM", "Center":"A", "Modality":"T1w", "nmi_warped_fixed":0.8},
        {"Method":"FewShot", "Center":"B", "Modality":"T2w", "nmi_warped_fixed":0.9},
    ])
    out = tmp_path / "figures"

    plot_violin(metrics_csv=csv, output_dir=out, x="modality", hue="method", aspect_ratio="16:9")

    assert (out / "nmi_warped_fixed.png").exists()

from registration_metrics.plot_violin import (
    canonical_feature_name,
    compute_group_positions,
    deduplicate_semantic_columns,
    infer_violin_width,
)


def test_task_and_modality_are_same_feature():
    assert canonical_feature_name("Task") == "modality"
    assert canonical_feature_name("task") == "modality"
    assert canonical_feature_name("Modality") == "modality"
    assert canonical_feature_name("modality") == "modality"


def test_deduplicate_semantic_columns():
    assert deduplicate_semantic_columns(["center", "modality", "task", "organ"]) == ["center", "modality", "organ"]


def test_shade_not_repeated_when_x_uses_modality():
    df = pd.DataFrame({
        "method": ["DDEM", "FewShot"],
        "center": ["A", "B"],
        "organ": ["Liver", "Kidney"],
        "modality": ["T1w", "T2w"],
        "task": ["Task1", "Task2"],
    })
    assert infer_shade_by(df, x_cols=["center", "organ", "modality"], hue="method", shade_by="auto") == []


def test_compute_group_positions_multi_group():
    positions = compute_group_positions(["A", "B"], {"A": ["m1", "m2"], "B": ["m1", "m2"]})
    assert positions[("A", "m2")] - positions[("A", "m1")] == 0.18
    center_a = (positions[("A", "m1")] + positions[("A", "m2")]) / 2
    center_b = (positions[("B", "m1")] + positions[("B", "m2")]) / 2
    assert round(center_b - center_a, 6) == 1.4


def test_compute_group_positions_single_group():
    positions = compute_group_positions(["A"], {"A": ["m1", "m2", "m3"]})
    vals = [positions[("A", key)] for key in ["m1", "m2", "m3"]]
    assert vals[1] == 0
    assert round(vals[0] + vals[2], 6) == 0
    assert min(vals) - 0.5 < min(vals)
    assert max(vals) + 0.5 > max(vals)


def test_violin_spacing_not_too_wide_multi_group():
    positions = compute_group_positions(["A", "B", "C"], {"A": ["m1", "m2"], "B": ["m1", "m2"], "C": ["m1", "m2"]})
    assert max(positions.values()) - min(positions.values()) <= 3.0
    assert infer_violin_width(2, 0.18) >= 0.15


def test_violin_spacing_not_too_crowded_single_group():
    positions = compute_group_positions(["A"], {"A": ["m1", "m2", "m3", "m4"]})
    vals = sorted(positions.values())
    gaps = np.diff(vals)
    assert np.allclose(gaps, 0.18)
    assert max(vals) - min(vals) > 0.5
