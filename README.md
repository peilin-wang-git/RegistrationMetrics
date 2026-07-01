# RegistrationMetrics

RegistrationMetrics 是一个用于跨模态医学图像配准指标计算与可视化的 Python 项目。它从多个按 `method / center / modality / task / organ` 分组的 CSV 中读取 fixed、moving、warped 图像、分割和 dense DVF 路径，输出 detailed metrics、combined metrics、frame-level motion metrics、case-level motion metrics、error log，并可生成论文风格 violin plot。

## 功能

- 全局图像指标：NMI、SSIM、LCC/NCC、MSE。NMI 使用 scikit-learn `normalized_mutual_info_score`（arithmetic average）定义，连续强度先用共享 bin edges 离散化，输出范围为 0 到 1。
- 分割指标：每个 label 和 foreground 的 Dice、IoU、HD95、ASSD；HD95/ASSD 使用 physical spacing，单位 mm。
- DVF 指标：Jacobian determinant min/max/mean/std、参与统计的 total_voxels、folding voxel 数量和 folding ratio。
- Organ ROI + NCCMove：对 liver、spleen、pancreas、kidney_left、kidney_right 和可选 kidney 执行 largest connected component bbox、equal range、iterative NCC ROI matching。
- Motion 指标：frame-level 和 case-level MovementError、AMD、RMSE、MAPE、PCC、AmplitudeAMD。
- VertebraNCC：从 multi-label segmentation 中的 vertebra labels 裁剪 ROI，计算 moving-fixed 和 warped-fixed NCC。
- 可视化：读取 combined 或 case-motion CSV，对 center、modality、method、organ 等分组输出 PNG/PDF/SVG violin plot，DPI=600，Times New Roman。

## 重要坐标说明

本项目不会硬编码 `array axis 0 = SI, axis 1 = AP, axis 2 = RL`。方向和 spacing 由 NIfTI affine 判断，输出标准 anatomical directions：AP、RL、SI。距离类指标均使用 physical spacing。若 affine 无法可靠判断方向，case 会被跳过并记录 `error_message`。

## Label map 注意事项

内置 label map 复现了用户提供的 multi-label 定义。其中 label 50 和 51 都是 `autochthon_right`，这可能是原始 label map 的重复或笔误；常见预期可能是 label 51 为 `clavicula_right`。用户可以在 YAML 中通过 `label_map` 覆盖：

```yaml
label_map:
  51: clavicula_right
```

## 空 mask 行为

- Dice/IoU：both empty 返回 1；one empty 返回 0。
- HD95/ASSD：both empty 返回 0；one empty 返回 NaN，并在日志中保留 skip/error 上下文。
- organ ROI 为空时不会中断全流程，而是记录 `organ_mask_empty` 并跳过该 organ 的 NCC matching。

## 配置示例

配置必须是 dictionary，而不是简单 list：

```yaml
CONFIG:
  DDEM:
    "Institution A | Liver | T1w-4D":
      center: "Institution A"
      modality: "T1w-4D"
      task: "T1w-4D"
      organ: "liver"
      csv_path: "/path/to/result.csv"
      output_dir: "./metrics_out"
```

每个输出行保留 `Method, Center, Modality, Task, Organ, AnalysisGroup, CaseID` 以及输入路径、Frame、status、error_message、skip_reason 和 runtime_seconds。

## CLI

```bash
python -m registration_metrics.cli compute \
  --config config.yaml \
  --output-dir ./metrics_out \
  --enable-global \
  --enable-seg \
  --enable-dvf \
  --enable-motion \
  --enable-vertebra
```

```bash
python -m registration_metrics.cli plot \
  --metrics-csv ./metrics_out/combined_metrics.csv \
  --case-motion-csv ./metrics_out/case_motion_metrics.csv \
  --output-dir ./figures \
  --hue center \
  --x modality
```

```bash
python -m registration_metrics.cli all \
  --config config.yaml \
  --output-dir ./metrics_out \
  --enable-global \
  --enable-seg \
  --enable-dvf \
  --enable-motion \
  --enable-vertebra \
  --plot
```

## 日志

项目使用 `logging` 输出非常详细的过程信息。console 输出 INFO，文件保存 DEBUG，日志文件名为 `metrics_run_YYYYMMDD_HHMMSS.log`。日志前缀包括：`[CONFIG]`、`[GROUP START]`、`[CASE START]`、`[LOAD]`、`[ORIENTATION]`、`[FRAME]`、`[GLOBAL]`、`[SEG]`、`[BOUND]`、`[EQUAL_RANGE]`、`[MATCH_NCC]`、`[MOTION FRAME]`、`[MOTION CASE]`、`[VERTEBRA]`、`[DVF]`、`[SAVE]`、`[ERROR]`、`[SUMMARY]`。

## Incremental saving and summary

During `compute`/`all`, each input CSV row is treated as one pre-split 3D case; 4D images or segmentations are skipped with `[SKIP NON-3D]`. Results are appended to `detailed_progress.csv` after each input CSV row finishes. Failures are appended to `error_log.csv` immediately. With `--num-workers > 1`, worker processes compute cases and the main process is the only writer for progress/error CSV files as futures complete. After processing completes, the program reloads `detailed_progress.csv` from disk and writes `summary_by_group.csv` and `summary_overall.csv`. Variance in summaries uses pandas `var` default `ddof=1`.

## GPU acceleration

CPU remains the default. Optional GPU acceleration can be requested with:

```bash
python -m registration_metrics.cli compute \
  --config config.yaml \
  --output-dir ./metrics_out \
  --enable-global \
  --enable-seg \
  --enable-dvf \
  --enable-motion \
  --enable-vertebra \
  --use-gpu \
  --device cuda:0 \
  --gpu-metrics all \
  --ncc-batch-size 64
```

Supported GPU paths include NCC/LCC, MSE, Dice, IoU, DVF Jacobian, VertebraNCC, and organ NCCMove/NCCMoveGT candidate NCC batches. GPU multiprocessing must use Python's `spawn` start method because Linux's default `fork` start method cannot safely re-initialize PyTorch CUDA in subprocesses. When `--use-gpu` and `--num-workers > 1` are set, the process pool is created with a spawn multiprocessing context and logs `[MP GPU] use_gpu=True num_workers=... start_method=spawn`. Multiprocessing with GPU can substantially increase GPU memory usage; for large images, start with `--num-workers 1` and only increase workers after verifying memory headroom. CPU multiprocessing may use multiple workers for throughput. NMI and SSIM remain CPU because the current implementations use numpy/sklearn binning and skimage. HD95/ASSD remain CPU because the distance transform uses scipy. If CUDA or a GPU metric fails, the code logs `[GPU FALLBACK]` and retries CPU for that metric.

## Dependencies

The NMI implementation requires `scikit-learn` because it intentionally matches `sklearn.metrics.normalized_mutual_info_score`.

## Segmentation organ metric scope

To reduce runtime and CSV width, individual Dice/IoU/HD95/ASSD columns are emitted only for the selected major heart-to-kidney organs by default:

```yaml
seg_metric_organs:
  - heart
  - liver
  - spleen
  - pancreas
  - kidney_left
  - kidney_right
  - stomach
  - aorta
  - inferior_vena_cava
```

The `mean_*_all_organs_*` column names are kept stable, but `all_organs` now means the default `SEG_MEAN_ORGANS` heart-to-kidney organ set, not every foreground label:

```yaml
seg_mean_organs:
  - heart
  - liver
  - spleen
  - pancreas
  - kidney_left
  - kidney_right
  - stomach
  - gallbladder
  - aorta
  - inferior_vena_cava
  - portal_vein_and_splenic_vein
  - duodenum
  - small_bowel
  - colon
```

Non-selected mean-only organs such as gallbladder, duodenum, small_bowel, and colon do not emit individual segmentation metric columns unless included in `seg_metric_organs`. Override individual-output organs with `seg_metric_organs` in YAML or `--seg-metric-organs ...`; override mean organs independently with `seg_mean_organs` in YAML or `--seg-mean-organs ...`; pass `--verbose-seg-mean` to debug-log every label decision for segmentation means.

## Motion mask quality fallback

For organ NCCMove/NCCMoveGT, masks with volume below `min_mask_volume_voxels` are treated as missing or too small, and severe pairwise volume mismatch below `severe_volume_ratio_threshold` marks the smaller side unreliable. Defaults are:

```yaml
min_mask_volume_voxels: 20
severe_volume_ratio_threshold: 0.20
```

When one side is unreliable and image geometry is compatible, the more reliable side's bbox is reused as the initial search bbox on the unreliable image before NCC matching. If both masks are invalid, NCCMove/NCCMoveGT displacement columns are set to NaN and diagnostic fallback columns record the skip reason.

## 3D volume metric mode

Metrics are volume-based for each 3D frame: LCC/NCC, MSE, NMI, SSIM, Dice, IoU, HD95, ASSD, organ-specific segmentation metrics, and mean segmentation metrics operate on full 3D volumes or 3D masks. For 4D inputs, the pipeline processes each frame as an independent 3D volume. SSIM uses skimage's n-dimensional `structural_similarity` on the full 3D volume with `channel_axis=None`, an automatically selected odd `win_size`, and a finite-voxel `data_range`; volumes too small for 3D SSIM return NaN with an `[SSIM SKIP]` log rather than falling back to 2D slice averaging.
