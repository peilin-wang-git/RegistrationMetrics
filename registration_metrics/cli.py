"""Command line interface for computing and plotting registration metrics."""
from __future__ import annotations
import argparse
from .config import load_config, setup_logging
from .io_utils import compute_from_config
from .motion_metrics import compute_frame_motion_metrics, compute_case_motion_metrics
from .plot_violin import plot_violin

def build_parser() -> argparse.ArgumentParser:
    """Build argparse CLI with compute, plot, and all subcommands."""
    p=argparse.ArgumentParser(prog="registration_metrics"); sub=p.add_subparsers(dest="cmd", required=True)
    def add_compute(s):
      s.add_argument("--config", required=True); s.add_argument("--output-dir", required=True); s.add_argument("--enable-global", action="store_true"); s.add_argument("--enable-seg", action="store_true"); s.add_argument("--enable-dvf", action="store_true"); s.add_argument("--enable-motion", action="store_true"); s.add_argument("--enable-vertebra", action="store_true"); s.add_argument("--use-gpu", action="store_true"); s.add_argument("--device", default="cuda:0"); s.add_argument("--gpu-metrics", default="all"); s.add_argument("--num-workers", type=int, default=1); s.add_argument("--ncc-batch-size", type=int, default=64); s.add_argument("--seg-metric-organs"); s.add_argument("--seg-mean-organs"); s.add_argument("--verbose-seg-mean", action="store_true"); s.add_argument("--min-mask-volume-voxels", type=int); s.add_argument("--severe-volume-ratio-threshold", type=float)
    add_compute(sub.add_parser("compute")); pl=sub.add_parser("plot"); pl.add_argument("--metrics-csv", nargs="+", required=True, help="One or more combined_metrics.csv / detailed_progress.csv files to plot together."); pl.add_argument("--case-motion-csv", nargs="*", default=None, help="Optional one or more case_motion_metrics.csv files to merge into plotting/statistics."); pl.add_argument("--output-dir", required=True); pl.add_argument("--hue", default=None, help="Primary color grouping column, or none to disable hue."); pl.add_argument("--x", default=None, help="X-axis metadata column or comma-separated columns, e.g. modality or center,organ,modality."); pl.add_argument("--shade-by", default="auto", help="Secondary shade grouping: auto, none, one column, or comma-separated columns."); pl.add_argument("--max-shade-levels", type=int, default=8, help="Warn when shade_group has more than this many levels."); pl.add_argument("--fig-width", type=float, help="Manual figure width in inches."); pl.add_argument("--fig-height", type=float, help="Manual figure height in inches."); pl.add_argument("--aspect-ratio", help="Figure aspect ratio such as 16:9, 4:3, or 3:2."); pl.add_argument("--auto-figure-size", dest="auto_figure_size", action="store_true", default=True, help="Automatically infer figure size from x groups, labels, and legend."); pl.add_argument("--no-auto-figure-size", dest="auto_figure_size", action="store_false", help="Disable automatic figure sizing when manually controlling size."); pl.add_argument("--min-fig-width", type=float, default=12.0); pl.add_argument("--max-fig-width", type=float, default=32.0); pl.add_argument("--base-fig-height", type=float, default=6.5); pl.add_argument("--legend-space-ratio", type=float, default=0.35); pl.add_argument("--save-statistics", action="store_true", help="Save descriptive statistics CSV files from the merged plot input."); pl.add_argument("--statistics-group-cols", default=None, help="Comma-separated group columns for statistics. Default: use available metadata columns: Method, Center, Modality, Task, Organ, AnalysisGroup."); pl.add_argument("--save-merged-plot-input", action="store_true", default=True, help="Save merged plot input CSV files to output directory.")
    al=sub.add_parser("all"); add_compute(al); al.add_argument("--plot", action="store_true"); al.add_argument("--hue", default="center"); al.add_argument("--x", default="modality")
    return p

def main(argv=None) -> None:
    """Run CLI."""
    args=build_parser().parse_args(argv); setup_logging(args.output_dir)
    if args.cmd in {"compute","all"}:
      cfg=load_config(args.config); combined,_,_=compute_from_config(cfg,args.output_dir,args.enable_global,args.enable_seg,args.enable_dvf,args.enable_motion,args.enable_vertebra,args.use_gpu,args.device,args.gpu_metrics,args.ncc_batch_size,args.seg_metric_organs,args.seg_mean_organs,args.verbose_seg_mean,args.min_mask_volume_voxels,args.severe_volume_ratio_threshold,args.num_workers)
      if args.enable_motion:
        f=compute_frame_motion_metrics(combined); f.to_csv(f"{args.output_dir}/frame_motion_metrics.csv", index=False)
        c=compute_case_motion_metrics(f); c.to_csv(f"{args.output_dir}/case_motion_metrics.csv", index=False)
      if args.cmd=="all" and args.plot: plot_violin(f"{args.output_dir}/combined_metrics.csv", f"{args.output_dir}/case_motion_metrics.csv", f"{args.output_dir}/figures", args.hue, args.x)
    elif args.cmd=="plot": plot_violin(args.metrics_csv,args.case_motion_csv,args.output_dir,args.hue,args.x,save_statistics=args.save_statistics,statistics_group_cols=args.statistics_group_cols,save_merged_plot_input=args.save_merged_plot_input,shade_by=args.shade_by,max_shade_levels=args.max_shade_levels,fig_width=args.fig_width,fig_height=args.fig_height,aspect_ratio=args.aspect_ratio,auto_figure_size=args.auto_figure_size,min_fig_width=args.min_fig_width,max_fig_width=args.max_fig_width,base_fig_height=args.base_fig_height,legend_space_ratio=args.legend_space_ratio)
if __name__ == "__main__": main()
