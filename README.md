# RegistrationMetrics

RegistrationMetrics 是一个用于跨模态医学图像配准指标计算与可视化的 Python 项目。它从多个按 `method / center / modality / task / organ` 分组的 CSV 中读取 fixed、moving、warped 图像、分割和 dense DVF 路径，输出 detailed metrics、combined metrics、frame-level motion metrics、case-level motion metrics、error log，并可生成论文风格 violin plot。

## 功能

- 全局图像指标：NMI、SSIM、LCC/NCC、MSE。
- 分割指标：每个 label 和 foreground 的 Dice、IoU、HD95、ASSD；HD95/ASSD 使用 physical spacing，单位 mm。
- DVF 指标：Jacobian determinant min/max/mean/std、folding voxel 数量和 folding ratio。
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
