import pandas as pd

from registration_metrics.io_utils import _build_tasks, deduplicate_manifest_rows


def test_deduplicate_manifest_by_case_id():
    df = pd.DataFrame({"case_id": ["a", "a"], "status": ["success", "skipped_complete"]})
    result = deduplicate_manifest_rows(df)
    assert len(result) == 1
    assert result.iloc[0]["status"] == "success"


def test_deduplicate_manifest_keep_latest_success():
    df = pd.DataFrame({
        "case_id": ["a", "a"], "status": ["success", "success"],
        "completed_at": ["2025-01-01T00:00:00Z", "2025-02-01T00:00:00Z"],
    })
    assert deduplicate_manifest_rows(df).iloc[0]["completed_at"] == "2025-02-01T00:00:00Z"


def test_deduplicate_manifest_without_case_id_uses_image_paths():
    df = pd.DataFrame({
        "fixed_img_path": ["f", "f"], "moving_img_path": ["m", "m"],
        "status": ["failed", "completed"],
    })
    result = deduplicate_manifest_rows(df)
    assert len(result) == 1
    assert result.iloc[0]["status"] == "completed"


def _build(config, output_dir):
    return _build_tasks(config, {}, [], [], False, False, False, False, False,
                        False, "cpu", 1, False, 1, 10.0, 64, output_dir)


def test_dedup_does_not_cross_groups(tmp_path):
    csv = tmp_path / "manifest.csv"
    pd.DataFrame({"case_id": ["a"], "status": ["success"]}).to_csv(csv, index=False)
    config = {
        "method_a": {"group": {"csv_path": str(csv)}},
        "method_b": {"group": {"csv_path": str(csv)}},
    }
    assert len(_build(config, tmp_path)) == 2


def test_tasks_are_unique_after_dedup(tmp_path):
    csv = tmp_path / "manifest.csv"
    pd.DataFrame({"case_id": ["a", "a"], "status": ["success", "skipped_complete"]}).to_csv(csv, index=False)
    config = {"method": {"group": {"csv_path": str(csv)}}}
    tasks = _build(config, tmp_path)
    assert len(tasks) == 1
    assert tasks[0].row_dict["status"] == "success"


def test_dedup_summary_files_saved(tmp_path):
    csv = tmp_path / "manifest.csv"
    pd.DataFrame({"case_id": ["a", "a"], "status": ["success", "failed"]}).to_csv(csv, index=False)
    _build({"method": {"group": {"csv_path": str(csv)}}}, tmp_path)
    assert (tmp_path / "manifest_dedup_summary.csv").exists()
    assert (tmp_path / "manifest_duplicates_removed.csv").exists()
    assert (tmp_path / "manifest_deduplicated_used_for_metrics.csv").exists()
    assert len(pd.read_csv(tmp_path / "manifest_duplicates_removed.csv")) == 1


def test_result_dedup_removes_old_duplicate_progress_rows(tmp_path):
    from registration_metrics.io_utils import _finalize_outputs

    progress = tmp_path / "detailed_progress.csv"
    pd.DataFrame({
        "Method": ["m", "m"], "Center": ["c", "c"], "Task": ["t", "t"],
        "Organ": ["o", "o"], "case_id": ["a", "a"], "metric": [1.0, 2.0],
    }).to_csv(progress, index=False)
    combined, _, _ = _finalize_outputs([], [], [], tmp_path, progress)
    assert len(combined) == 1
    assert combined.iloc[0]["metric"] == 2.0
    assert len(pd.read_csv(tmp_path / "combined_metrics.csv")) == 1
    assert len(pd.read_csv(progress)) == 1
