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
      s.add_argument("--config", required=True); s.add_argument("--output-dir", required=True); s.add_argument("--enable-global", action="store_true"); s.add_argument("--enable-seg", action="store_true"); s.add_argument("--enable-dvf", action="store_true"); s.add_argument("--enable-motion", action="store_true"); s.add_argument("--enable-vertebra", action="store_true")
    add_compute(sub.add_parser("compute")); pl=sub.add_parser("plot"); pl.add_argument("--metrics-csv"); pl.add_argument("--case-motion-csv"); pl.add_argument("--output-dir", required=True); pl.add_argument("--hue", default="center"); pl.add_argument("--x", default="modality")
    al=sub.add_parser("all"); add_compute(al); al.add_argument("--plot", action="store_true"); al.add_argument("--hue", default="center"); al.add_argument("--x", default="modality")
    return p

def main(argv=None) -> None:
    """Run CLI."""
    args=build_parser().parse_args(argv); setup_logging(args.output_dir)
    if args.cmd in {"compute","all"}:
      cfg=load_config(args.config); combined,_,_=compute_from_config(cfg,args.output_dir,args.enable_global,args.enable_seg,args.enable_dvf,args.enable_motion,args.enable_vertebra)
      if args.enable_motion:
        f=compute_frame_motion_metrics(combined); f.to_csv(f"{args.output_dir}/frame_motion_metrics.csv", index=False)
        c=compute_case_motion_metrics(f); c.to_csv(f"{args.output_dir}/case_motion_metrics.csv", index=False)
      if args.cmd=="all" and args.plot: plot_violin(f"{args.output_dir}/combined_metrics.csv", f"{args.output_dir}/case_motion_metrics.csv", f"{args.output_dir}/figures", args.hue, args.x)
    elif args.cmd=="plot": plot_violin(args.metrics_csv,args.case_motion_csv,args.output_dir,args.hue,args.x)
if __name__ == "__main__": main()
