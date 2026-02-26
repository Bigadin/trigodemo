#!/usr/bin/env python3
"""
Optimise les vidéos du dataset (réduction résolution + compression H.264).

Usage:
    python scripts/optimize_videos.py [--dry-run] [--min-mb 25]

Par défaut optimise les vidéos > 25 MB.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
VIDEOS_DIR = ROOT / "videos"
MAX_WIDTH = 1280
MAX_HEIGHT = 720
CRF = 28  # Qualité H.264 (18-28 = bon compromis)


def optimize_video(src: Path, dst: Path) -> bool:
    """Réencode avec ffmpeg pour réduire la taille."""
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(src),
                "-vf", f"scale='min({MAX_WIDTH},iw)':'min({MAX_HEIGHT},ih)':force_original_aspect_ratio=decrease",
                "-c:v", "libx264", "-crf", str(CRF),
                "-preset", "fast", "-c:a", "aac", "-b:a", "128k", "-f", "mp4",
                str(dst)
            ],
            capture_output=True,
            timeout=600,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"  Erreur: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Optimise les vidéos du dataset")
    parser.add_argument("--dry-run", action="store_true", help="Afficher sans modifier")
    parser.add_argument("--min-mb", type=float, default=25, help="Optimiser si > N MB (défaut: 25)")
    parser.add_argument("--all", action="store_true", help="Optimiser toutes les vidéos")
    args = parser.parse_args()

    if not VIDEOS_DIR.exists():
        print(f"Dossier {VIDEOS_DIR} introuvable.")
        sys.exit(1)

    videos = list(VIDEOS_DIR.glob("*.mp4")) + list(VIDEOS_DIR.glob("*.webm")) + list(VIDEOS_DIR.glob("*.mov"))
    if not videos:
        print("Aucune vidéo trouvée.")
        sys.exit(0)

    to_optimize = []
    for v in sorted(videos):
        size_mb = v.stat().st_size / (1024 * 1024)
        if args.all or size_mb > args.min_mb:
            to_optimize.append((v, size_mb))

    if not to_optimize:
        print(f"Toutes les vidéos font < {args.min_mb} MB. Rien à optimiser.")
        sys.exit(0)

    print(f"Vidéos à optimiser ({len(to_optimize)}):")
    for v, mb in to_optimize:
        print(f"  - {v.name}: {mb:.1f} MB")

    if args.dry_run:
        print("\n[--dry-run] Aucune modification.")
        sys.exit(0)

    for src, size_before in to_optimize:
        stem = src.stem
        tmp = VIDEOS_DIR / (stem + "_tmp_opt.mp4")
        out = VIDEOS_DIR / (stem + ".mp4")
        print(f"\nOptimisation de {src.name}...")
        if optimize_video(src, tmp):
            size_after = tmp.stat().st_size / (1024 * 1024)
            gain = 100 * (1 - size_after / size_before) if size_before > 0 else 0
            src.unlink()
            tmp.rename(out)
            print(f"  OK: {size_before:.1f} MB -> {size_after:.1f} MB (-{gain:.0f}%)")
        else:
            tmp.unlink(missing_ok=True)
            print(f"  ÉCHEC (ffmpeg requis)")

    print("\nTerminé.")


if __name__ == "__main__":
    main()
