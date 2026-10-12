"""
setup.py - Automated asset downloader for iris_core.
Downloads and sets up:
  1. Binaries: llama-server.exe (b10587) and parakeet-server.exe (v0.5.0) in core/bin/
  2. Speech & Language Models: Ling-3.0-tiny-Q4_0.gguf and tdt_ctc-110m-f16.gguf in core/modal/
  3. TTS Weights: kokoro_en_7m.pth and af_msa.pt in core/tts/
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent
CORE_DIR = ROOT_DIR / "core"
BIN_DIR = CORE_DIR / "bin"
MODELS_DIR = CORE_DIR / "modal"
TTS_DIR = CORE_DIR / "tts"

ASSETS = {
    "bin": [
        {
            "name": "llama-server (b10587 win-cpu-x64)",
            "url": "https://github.com/ggml-org/llama.cpp/releases/download/b10587/llama-b10587-bin-win-cpu-x64.zip",
            "target_dir": BIN_DIR / "llama-b10587-bin-win-cpu-x64",
            "verify_file": "llama-server.exe",
            "is_zip": True,
        },
        {
            "name": "parakeet-server (v0.5.0 win-cpu-x64)",
            "url": "https://github.com/mudler/parakeet.cpp/releases/download/v0.5.0/parakeet-v0.5.0-bin-win-cpu-x64.zip",
            "target_dir": BIN_DIR / "parakeet-v0.5.0-bin-win-cpu-x64",
            "verify_file": "parakeet-server.exe",
            "is_zip": True,
        },
    ],
    "models": [
        {
            "name": "Parakeet STT GGUF (tdt_ctc-110m-f16)",
            "url": "https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/main/tdt_ctc-110m-f16.gguf",
            "target_path": MODELS_DIR / "tdt_ctc-110m-f16.gguf",
            "expected_size": 267452544,
        },
        {
            "name": "Ling-3.0-tiny MoE GGUF (Q4_0, 1.3B active)",
            "url": "https://huggingface.co/bloomer010/Ling-3.0-tiny-GGUF/resolve/main/Ling-3.0-tiny-Q4_0.gguf",
            "target_path": MODELS_DIR / "Ling-3.0-tiny-Q4_0.gguf",
            "expected_size": 4526279744,
        },
    ],
    "tts": [
        {
            "name": "Kokoro-7M Distilled Weights",
            "url": "https://huggingface.co/oddadmix/Kokoro-7M-Distill/resolve/main/kokoro_en_7m.pth",
            "target_path": TTS_DIR / "kokoro_en_7m.pth",
            "expected_size": 30096222,
        },
        {
            "name": "Kokoro Voice Pack (af_msa.pt)",
            "url": "https://huggingface.co/oddadmix/Kokoro-7M-Distill/resolve/main/af_msa.pt",
            "target_path": TTS_DIR / "af_msa.pt",
            "expected_size": 523887,
        },
    ],
}


def download_file(url: str, dest_path: Path, label: str) -> bool:
    """Download file with live progress indicator."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".downloading")

    print(f"📥 Downloading {label}...")
    print(f"   Source: {url}")
    print(f"   Target: {dest_path}")

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) iris-core/1.0"}
    req = urllib.request.Request(url, headers=headers)

    t0 = time.perf_counter()
    downloaded = 0
    last_print = 0

    try:
        with urllib.request.urlopen(req) as resp, open(temp_path, "wb") as out_file:
            total_size = int(resp.headers.get("content-length", 0))

            while True:
                chunk = resp.read(1024 * 1024)  # 1MB buffer
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)

                now = time.perf_counter()
                if now - last_print > 0.4:
                    last_print = now
                    elapsed = max(now - t0, 0.001)
                    speed_mb = (downloaded / (1024 * 1024)) / elapsed
                    if total_size > 0:
                        pct = (downloaded / total_size) * 100
                        print(
                            f"\r   [{pct:5.1f}%] {downloaded / (1024*1024):.1f} MB / "
                            f"{total_size / (1024*1024):.1f} MB ({speed_mb:.1f} MB/s)",
                            end="",
                            flush=True,
                        )
                    else:
                        print(
                            f"\r   {downloaded / (1024*1024):.1f} MB downloaded ({speed_mb:.1f} MB/s)",
                            end="",
                            flush=True,
                        )

        print(f"\r   [100.0%] {downloaded / (1024*1024):.1f} MB complete! ({time.perf_counter() - t0:.1f}s)")
        if temp_path.exists():
            if dest_path.exists():
                dest_path.unlink()
            temp_path.replace(dest_path)
        return True

    except Exception as e:
        print(f"\n❌ Failed to download {label}: {e}")
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass
        return False


def setup_binaries(force: bool = False) -> bool:
    """Download and extract server binaries."""
    print("\n" + "=" * 70)
    print("📦 SETTING UP SERVER BINARIES (llama-server & parakeet-server)")
    print("=" * 70)
    all_ok = True

    for item in ASSETS["bin"]:
        target_dir = item["target_dir"]
        verify_path = target_dir / item["verify_file"]

        if verify_path.exists() and not force:
            print(f"✓ {item['name']} already installed at: {verify_path}")
            continue

        zip_temp = BIN_DIR / f"{target_dir.name}.zip"
        ok = download_file(item["url"], zip_temp, item["name"])
        if not ok:
            all_ok = False
            continue

        print(f"📂 Extracting {zip_temp.name} to {target_dir}...")
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_temp, "r") as z:
                z.extractall(target_dir)
            if zip_temp.exists():
                zip_temp.unlink()
            print(f"✓ Extracted successfully: {verify_path}")
        except Exception as e:
            print(f"❌ Failed to extract {zip_temp}: {e}")
            all_ok = False

    return all_ok


def setup_models(force: bool = False) -> bool:
    """Download GGUF language and speech models."""
    print("\n" + "=" * 70)
    print("🧠 SETTING UP GGUF MODELS (Ling-3.0-tiny & Parakeet STT)")
    print("=" * 70)
    all_ok = True

    for item in ASSETS["models"]:
        path: Path = item["target_path"]
        expected: int = item.get("expected_size", 0)

        if path.exists() and not force:
            actual = path.stat().st_size
            if expected == 0 or actual == expected or actual > expected * 0.95:
                print(f"✓ {item['name']} already present ({actual / (1024*1024):.1f} MB): {path}")
                continue

        ok = download_file(item["url"], path, item["name"])
        if not ok:
            all_ok = False

    return all_ok


def setup_tts(force: bool = False) -> bool:
    """Download Kokoro TTS distilled weights."""
    print("\n" + "=" * 70)
    print("🔊 SETTING UP KOKORO-7M TTS WEIGHTS")
    print("=" * 70)
    all_ok = True

    for item in ASSETS["tts"]:
        path: Path = item["target_path"]
        expected: int = item.get("expected_size", 0)

        if path.exists() and not force:
            actual = path.stat().st_size
            if expected == 0 or actual == expected:
                print(f"✓ {item['name']} already present ({actual / (1024*1024):.1f} MB): {path}")
                continue

        ok = download_file(item["url"], path, item["name"])
        if not ok:
            all_ok = False

    return all_ok


def check_status() -> None:
    """Check asset installation status."""
    print("\n" + "=" * 70)
    print("🔍 CHECKING IRIS_CORE ASSET STATUS")
    print("=" * 70)

    # Binaries
    print("\n[Server Binaries]")
    for item in ASSETS["bin"]:
        vpath = item["target_dir"] / item["verify_file"]
        status = "✅ INSTALLED" if vpath.exists() else "❌ MISSING"
        print(f"  {status:12} {item['name']} -> {vpath}")

    # Models
    print("\n[Speech & Language Models]")
    for item in ASSETS["models"]:
        mpath = item["target_path"]
        if mpath.exists():
            sz = mpath.stat().st_size / (1024 * 1024)
            status = f"✅ PRESENT ({sz:.1f} MB)"
        else:
            status = "❌ MISSING"
        print(f"  {status:24} {item['name']} -> {mpath}")

    # TTS
    print("\n[Kokoro TTS Weights]")
    for item in ASSETS["tts"]:
        tpath = item["target_path"]
        if tpath.exists():
            sz = tpath.stat().st_size / (1024 * 1024)
            status = f"✅ PRESENT ({sz:.1f} MB)"
        else:
            status = "❌ MISSING"
        print(f"  {status:24} {item['name']} -> {tpath}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Automated setup for iris_core models and binaries.")
    parser.add_argument("--check", action="store_true", help="Check status of all required assets without downloading.")
    parser.add_argument("--bin", action="store_true", help="Download and set up only server binaries.")
    parser.add_argument("--models", "--modal", action="store_true", help="Download only GGUF speech & language models.")
    parser.add_argument("--tts", action="store_true", help="Download only Kokoro TTS weights.")
    parser.add_argument("--force", action="store_true", help="Force re-download even if files already exist.")

    args = parser.parse_args()

    if args.check:
        check_status()
        return

    # If no specific group selected, set up all
    do_all = not (args.bin or args.models or args.tts)

    success = True
    if do_all or args.bin:
        success = setup_binaries(force=args.force) and success

    if do_all or args.models:
        success = setup_models(force=args.force) and success

    if do_all or args.tts:
        success = setup_tts(force=args.force) and success

    check_status()

    if success:
        print("\n🎉 All requested iris_core assets are ready!")
        print("You can now start the companion with: python main.py")
    else:
        print("\n⚠️ Some assets failed to download. Please check the logs above and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
