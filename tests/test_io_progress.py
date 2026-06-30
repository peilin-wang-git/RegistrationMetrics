import pandas as pd
from registration_metrics.io_utils import append_rows_to_csv, save_summary_from_progress


def test_append_rows_to_csv_header_once(tmp_path):
    path = tmp_path / "detailed_progress.csv"
    append_rows_to_csv([{"case_id":"a", "Method":"M", "Center":"C", "Modality":"T", "Task":"T", "Organ":"liver", "metric_x":1.0}], path)
    append_rows_to_csv([{"case_id":"b", "Method":"M", "Center":"C", "Modality":"T", "Task":"T", "Organ":"liver", "metric_x":3.0}], path)
    text = path.read_text()
    assert text.count("case_id") == 1
    assert len(pd.read_csv(path)) == 2


def test_summary_reloads_progress_csv(tmp_path):
    path = tmp_path / "detailed_progress.csv"
    append_rows_to_csv([
        {"case_id":"a", "Method":"M", "Center":"C", "Modality":"T", "Task":"T", "Organ":"liver", "metric_x":1.0},
        {"case_id":"b", "Method":"M", "Center":"C", "Modality":"T", "Task":"T", "Organ":"liver", "metric_x":3.0},
    ], path)
    group, overall = save_summary_from_progress(path, tmp_path)
    assert (tmp_path / "summary_by_group.csv").exists()
    assert (tmp_path / "summary_overall.csv").exists()
    assert not group.empty
    assert float(overall.loc[overall["metric"] == "metric_x", "mean"].iloc[0]) == 2.0
