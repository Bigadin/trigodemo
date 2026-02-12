"""
Build script – generates a self-contained dist/ folder with minified assets.

Usage:
    python build.py

The output in dist/ can be served with any static file server:
    python -m http.server 3000 --directory dist
"""

import shutil
import gzip
from pathlib import Path

# --------------- config ---------------
ROOT = Path(__file__).parent
SRC_STATIC = ROOT / "static"
DIST = ROOT / "dist"

# Files / dirs to copy
COPY_DIRS = ["assets_youn"]
EXTRA_FILES = ["observations.html"]  # keep secondary pages if any

# --------------- helpers ---------------

def _minify_css(text: str) -> str:
    """Minify CSS – try csscompressor, fallback to basic strip."""
    try:
        import csscompressor
        return csscompressor.compress(text)
    except Exception:
        # basic fallback: remove comments + collapse whitespace
        import re
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.S)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()


def _minify_js(text: str) -> str:
    """Minify JS – try rjsmin, fallback to basic strip."""
    try:
        import rjsmin
        return rjsmin.jsmin(text)
    except Exception:
        # basic fallback
        import re
        text = re.sub(r'//[^\n]*', '', text)
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.S)
        text = re.sub(r'\n\s*\n', '\n', text)
        return text.strip()


def _gzip_file(path: Path):
    """Create a .gz pre-compressed sibling."""
    gz_path = path.with_suffix(path.suffix + '.gz')
    with open(path, 'rb') as f_in:
        with gzip.open(gz_path, 'wb', compresslevel=9) as f_out:
            f_out.write(f_in.read())


def _size_str(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    return f"{n / 1024:.1f} KB"

# --------------- main ---------------

def build():
    print("=== YRYS UI – Build ===\n")

    # Clean
    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir()
    (DIST / "css").mkdir()
    (DIST / "js").mkdir()

    # ---- HTML ----
    html_src = SRC_STATIC / "index.html"
    html = html_src.read_text(encoding="utf-8")
    # Rewrite asset paths: /static/... → ./...
    html = html.replace('/static/', './')
    # Also handle src="/static/... and href="/static/...
    html = html.replace('="/static/', '="./')
    (DIST / "index.html").write_text(html, encoding="utf-8")
    print(f"  index.html  {_size_str(html_src.stat().st_size)}")

    # ---- CSS ----
    css_src = SRC_STATIC / "css" / "index.css"
    css_raw = css_src.read_text(encoding="utf-8")
    css_min = _minify_css(css_raw)
    (DIST / "css" / "index.css").write_text(css_min, encoding="utf-8")
    _gzip_file(DIST / "css" / "index.css")
    orig = css_src.stat().st_size
    mini = len(css_min.encode("utf-8"))
    gz = (DIST / "css" / "index.css.gz").stat().st_size
    print(f"  index.css   {_size_str(orig)} -> {_size_str(mini)} (min) -> {_size_str(gz)} (gz)")

    # ---- JS ----
    for js_file in (SRC_STATIC / "js").glob("*.js"):
        js_raw = js_file.read_text(encoding="utf-8")
        js_min = _minify_js(js_raw)
        dest = DIST / "js" / js_file.name
        dest.write_text(js_min, encoding="utf-8")
        _gzip_file(dest)
        orig = js_file.stat().st_size
        mini = len(js_min.encode("utf-8"))
        gz = dest.with_suffix(dest.suffix + '.gz').stat().st_size
        print(f"  {js_file.name:<20s} {_size_str(orig)} -> {_size_str(mini)} (min) -> {_size_str(gz)} (gz)")

    # ---- Assets (SVG, images) ----
    for d in COPY_DIRS:
        src_dir = SRC_STATIC / d
        if src_dir.exists():
            shutil.copytree(src_dir, DIST / d, dirs_exist_ok=True)
            count = sum(1 for _ in (DIST / d).rglob("*") if _.is_file())
            print(f"  {d}/  ({count} files copied)")

    # ---- Extra files ----
    for f in EXTRA_FILES:
        src = SRC_STATIC / f
        if src.exists():
            shutil.copy2(src, DIST / f)
            print(f"  {f}  copied")

    # ---- Summary ----
    total = sum(f.stat().st_size for f in DIST.rglob("*") if f.is_file())
    total_no_gz = sum(f.stat().st_size for f in DIST.rglob("*") if f.is_file() and not f.name.endswith('.gz'))
    print(f"\n  Total dist/  {_size_str(total_no_gz)} (sans gz) / {_size_str(total)} (avec gz)")
    print(f"\n  Servir avec:")
    print(f"    python -m http.server 3000 --directory dist")
    print(f"    puis ouvrir http://localhost:3000\n")


if __name__ == "__main__":
    build()
