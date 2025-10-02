# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['src\\fire_detector.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('data/models/best.pt', 'data/models'),       # YOLO model
        ('data/siren/fire-alarm.mp3', 'data/siren')   # Siren audio
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='fire_detector',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='fire_detector',
)
