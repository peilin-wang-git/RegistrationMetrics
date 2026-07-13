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
