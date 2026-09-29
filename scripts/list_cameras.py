"""List connected RealSense cameras and flag which one recorded the datasets.

The model is camera-aware and was trained on captures from one specific unit,
so it matters which physical camera is plugged in. Serial numbers are printed
by `rs-enumerate-devices` too, but this also cross-checks them against the
serials recorded in the datasets' session.json files.

    python3 scripts/list_cameras.py
"""
import json
from pathlib import Path

import pyrealsense2 as rs


def dataset_serials():
    """Serial -> list of datasets recorded with it, read from session.json."""
    found = {}
    for sj in Path('.').glob('*/**/session.json'):
        try:
            sn = json.loads(sj.read_text()).get('camera_serial_number')
        except (json.JSONDecodeError, OSError):
            continue
        if sn:
            found.setdefault(str(sn), set()).add(sj.parts[0])
    return {k: sorted(v) for k, v in found.items()}


def main():
    known = dataset_serials()
    print("Серийники, которыми сняты датасеты:")
    if known:
        for sn, dsets in known.items():
            print(f"  {sn}  <- {', '.join(dsets)}")
    else:
        print("  (в проекте не найдено ни одного session.json)")

    devs = list(rs.context().query_devices())
    print(f"\nПодключено камер: {len(devs)}")
    if not devs:
        print("  Ничего не найдено. Проверьте USB-кабель (нужен USB 3.x) и питание.")
        return

    for i, d in enumerate(devs):
        sn = d.get_info(rs.camera_info.serial_number)
        mark = "СОВПАДАЕТ с датасетом" if sn in known else "другая камера"
        print(f"\n  [{i}] {d.get_info(rs.camera_info.name)}")
        print(f"      serial   : {sn}   <-- {mark}")
        print(f"      firmware : {d.get_info(rs.camera_info.firmware_version)}")
        print(f"      USB      : {d.get_info(rs.camera_info.usb_type_descriptor)}")
        try:
            print(f"      physical : {d.get_info(rs.camera_info.physical_port)}")
        except RuntimeError:
            pass

        # Intrinsics come from a live stream profile, so this needs the camera
        # briefly opened rather than just enumerated.
        try:
            pipe, conf = rs.pipeline(), rs.config()
            conf.enable_device(sn)
            conf.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            prof = pipe.start(conf)
            ci = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            pipe.stop()
            print(f"      intrinsics: fx={ci.fx:.1f} fy={ci.fy:.1f} "
                  f"cx={ci.ppx:.1f} cy={ci.ppy:.1f} @ {ci.width}x{ci.height}")
        except RuntimeError as e:
            print(f"      intrinsics: не удалось открыть поток ({e})")


if __name__ == '__main__':
    main()
