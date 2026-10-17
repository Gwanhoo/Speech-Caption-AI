# -*- mode: python ; coding: utf-8 -*-
"""Minimal Windows client build for the Phase 5 VOXLUX GUI.

This intentionally packages only the desktop client path.  Phase 4's GPU
server, separation libraries, and local model checkpoints stay outside the
Windows client distribution.
"""

from pathlib import Path


project_root = Path(SPECPATH)

a = Analysis(
    [str(project_root / "phase5" / "gui_app.py")],
    pathex=[
        str(project_root),
        str(project_root / "phase3"),
        str(project_root / "phase4"),
        str(project_root / "phase5"),
    ],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "gpu_processing_server",
        "clearvoice",
        "faster_whisper",
        "funasr",
        "silero_vad",
        "torch",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="VOXLUX",
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="VOXLUX",
)
