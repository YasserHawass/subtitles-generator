#!/usr/bin/env python3
"""
subgen.py — Simple CLI to generate SRT subtitles for a single video or a directory of videos.

Dependencies (install once):
    pip install faster-whisper==1.* ffmpeg-python==0.* # ffmpeg must be installed on your system PATH
    conda install -c nvidia/label/cuda-12.4.0 cudnn
Model requirement:
    - This script can now automatically download the model from Hugging Face Hub.
    - Set DEFAULT_MODEL below to a model name like "base", "small", "medium", or "large-v3".

Usage examples:
    # Single file, with default model
    python subgen.py /path/to/video.mp4

    # Single file, with a specific model
    python subgen.py /path/to/video.mp4 --model medium

    # Directory, recursive, skip first 10 files
    python subgen.py /path/to/videos --dir --recursive --skip 10

    # Custom output directory and overwrite existing SRTs
    python subgen.py /path/to/videos --dir -o /tmp/subs --overwrite

    # Dry-run to preview which files would be processed
    python subgen.py /path/to/videos --dir --dry-run

Exit codes:
    0  Success (all requested files processed)
    1  Some files failed (see log)
    2  Fatal argument or environment error
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

# -------------------------
# HARD-CODED MODEL LOCATION (now a name for automatic download)
# -------------------------
# This model will be automatically downloaded from Hugging Face Hub if not found locally.
# Choose a model name like "tiny", "base", "small", "medium", or "large-v3".
DEFAULT_MODEL = "base"


# -------------------------
# Logging configuration
# -------------------------
LOGGER = logging.getLogger("subgen")
HANDLER = logging.StreamHandler()
FORMATTER = logging.Formatter("[%(levelname)s] %(message)s")
HANDLER.setFormatter(FORMATTER)
LOGGER.addHandler(HANDLER)
LOGGER.setLevel(logging.INFO)


# -------------------------
# Utilities
# -------------------------
VIDEO_EXTS = {
    ".mp4", ".mkv", ".mov", ".avi", ".m4v", ".webm",
    ".wmv", ".flv", ".mpg", ".mpeg", ".mp3", ".wav", ".m4a", ".aac", ".ogg",
}

def is_media_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in VIDEO_EXTS

def srt_timestamp(seconds: float) -> str:
    """
    Convert seconds (float) to SRT timestamp "HH:MM:SS,mmm".
    The SRT standard requires comma for milliseconds and fixed-width fields.
    """
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000.0))
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def default_srt_path(video_path: Path, out_dir: Optional[Path]) -> Path:
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / (video_path.stem + ".srt")
    return video_path.with_suffix(".srt")


# -------------------------
# Model setup
# -------------------------
@dataclass
class ModelConfig:
    model_name: str
    device: str = "auto"
    compute_type: str = "auto"
    vad_filter: bool = True
    language: Optional[str] = None
    beam_size: int = 5
    temperature: float = 0.0

def init_model(cfg: ModelConfig):
    """
    Load faster-whisper model from a model name. The model will be
    automatically downloaded from Hugging Face Hub if it's not present.
    """
    try:
        from faster_whisper import WhisperModel
    except Exception as e:
        LOGGER.error("faster-whisper is not installed. Run: pip install faster-whisper")
        raise SystemExit(2) from e

    LOGGER.info("Loading model '%s'. Downloading if not in cache.", cfg.model_name)
    try:
        model = WhisperModel(
            cfg.model_name,
            device=cfg.device,
            compute_type=cfg.compute_type,
        )
        return model
    except Exception as e:
        LOGGER.error("Failed to load or download model '%s': %s", cfg.model_name, e)
        raise SystemExit(2) from e


# -------------------------
# Core transcription
# -------------------------
def process_video(
    video_path: Path,
    model,
    cfg: ModelConfig,
) -> List[Tuple[float, float, str]]:
    """
    Transcribe a single media file.
    Returns a list of (start_sec, end_sec, text).
    """
    LOGGER.info("Transcribing: %s", video_path)
    segments_iter, _info = model.transcribe(
        str(video_path),
        language=cfg.language,
        vad_filter=cfg.vad_filter,
        beam_size=cfg.beam_size,
        temperature=cfg.temperature,
    )
    segments: List[Tuple[float, float, str]] = []
    for seg in segments_iter:
        segments.append((float(seg.start or 0.0), float(seg.end or 0.0), seg.text.strip()))
    return segments


def save_subtitles(
    segments: List[Tuple[float, float, str]],
    out_path: Path,
    overwrite: bool = False,
) -> Path:
    """
    Write SRT file.
    """
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"SRT already exists (use --overwrite to replace): {out_path}")

    lines: List[str] = []
    for i, (start, end, text) in enumerate(segments, start=1):
        if not text:
            continue
        start_ts = srt_timestamp(start)
        end_ts = srt_timestamp(end)
        lines.append(f"{i}")
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(text)
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("Wrote: %s", out_path)
    return out_path


# -------------------------
# Batch / directory handling
# -------------------------
def iter_media_files(root: Path, recursive: bool) -> Iterable[Path]:
    if root.is_file() and is_media_file(root):
        yield root
        return
    if root.is_dir():
        if recursive:
            for p in sorted(root.rglob("*")):
                if is_media_file(p):
                    yield p
        else:
            for p in sorted(root.iterdir()):
                if is_media_file(p):
                    yield p

def run_batch(
    input_path: Path,
    model,
    cfg: ModelConfig,
    as_directory: bool,
    recursive: bool,
    skip: int,
    out_dir: Optional[Path],
    overwrite: bool,
    dry_run: bool,
) -> Tuple[int, int]:
    """
    Returns (ok_count, fail_count).
    """
    files: List[Path] = []

    if as_directory:
        files = list(iter_media_files(input_path, recursive=recursive))
        if skip > 0:
            files = files[skip:]
        if not files:
            LOGGER.warning("No media files found to process.")
            return (0, 0)
    else:
        if not input_path.exists():
            LOGGER.error("Input file not found: %s", input_path)
            return (0, 1)
        if not is_media_file(input_path):
            LOGGER.error("Input path is not a supported media file: %s", input_path)
            return (0, 1)
        files = [input_path]

    ok = 0
    failed = 0

    for vid in files:
        try:
            out_srt = default_srt_path(vid, out_dir)
            if dry_run:
                LOGGER.info("[DRY-RUN] Would process: %s -> %s", vid, out_srt)
                ok += 1
                continue

            segments = process_video(vid, model, cfg)
            save_subtitles(segments, out_srt, overwrite=overwrite)
            ok += 1
        except FileExistsError as e:
            LOGGER.error(str(e))
            failed += 1
        except KeyboardInterrupt:
            LOGGER.error("Interrupted by user.")
            break
        except Exception as e:
            LOGGER.exception("Failed: %s", vid)
            failed += 1

    return ok, failed


# -------------------------
# CLI
# -------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="subgen",
        description="Generate SRT subtitles for a video or a directory of videos using a local Whisper model (faster-whisper).",
    )
    p.add_argument(
        "input",
        type=str,
        help="Path to a video/audio file or a directory (use --dir for directory mode).",
    )
    p.add_argument(
        "--dir",
        action="store_true",
        help="Treat the input path as a directory and process its media files.",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories when used with --dir.",
    )
    p.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Skip the first N files when processing a directory.",
    )
    p.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Directory to write SRT files into (mirrors flat list; no tree recreation). Defaults to alongside each video.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .srt files if they already exist.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List the files that would be processed without doing any work.",
    )
    p.add_argument(
        "--language",
        type=str,
        default=None,
        help="Force language code (e.g., 'en', 'ar'). Default: auto-detect.",
    )
    p.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable VAD (voice activity detection) segmentation.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device for faster-whisper. Default: auto.",
    )
    p.add_argument(
        "--compute-type",
        type=str,
        default="auto",
        help="Compute type, e.g., 'int8', 'int8_float16', 'float16', or 'auto'. Default: auto.",
    )
    p.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"The model name to use from Hugging Face Hub (e.g., 'large-v3'). Default: '{DEFAULT_MODEL}'.",
    )
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging.",
    )
    return p

def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        LOGGER.setLevel(logging.DEBUG)

    input_path = Path(args.input).expanduser()
    out_dir = Path(args.output_dir).expanduser() if args.output_dir else None

    cfg = ModelConfig(
        model_name=args.model,
        device=args.device,
        compute_type=args.compute_type,
        vad_filter=not args.no_vad,
        language=args.language,
    )

    if args.dry_run and args.dir:
        model = None
    else:
        model = init_model(cfg)

    ok, failed = run_batch(
        input_path=input_path,
        model=model,
        cfg=cfg,
        as_directory=bool(args.dir),
        recursive=bool(args.recursive),
        skip=int(args.skip),
        out_dir=out_dir,
        overwrite=bool(args.overwrite),
        dry_run=bool(args.dry_run),
    )

    if failed > 0:
        LOGGER.warning("Done with issues: %d succeeded, %d failed.", ok, failed)
        return 1 if ok > 0 else 1
    LOGGER.info("Done: %d file(s) processed successfully.", ok)
    return 0


if __name__ == "__main__":
    sys.exit(main())