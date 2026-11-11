#!/bin/zsh
# Flash the firmware built by build_tars.sh: bootloader, partition table, otadata, app and assets (exactly what `idf.py flash` writes).
# Does NOT erase the whole flash, so the servo zero calibration (flash 0xFFF000), Wi-Fi and other NVS settings stay.
# usage: tools/tars_log/flash_tars.sh /dev/cu.usbmodemXXXX
set -e
PORT="$1"
[ -z "$PORT" ] && { echo "usage: $0 /dev/cu.usbmodemXXXX   (ls /dev/cu.usbmodem*)"; exit 1; }
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${IDF_PYTHON:-$(ls -d $HOME/.espressif/python_env/idf5.5_py*_env/bin/python 2>/dev/null | tail -1)}"
[ -z "$PY" ] && PY=python3
cd "$ROOT/build/packages/tars-cn"
[ -f flash_args ] || { echo "no build found: run tools/tars_log/build_tars.sh first"; exit 1; }
"$PY" -m esptool --chip esp32s3 -p "$PORT" -b 460800 --before default_reset --after hard_reset write_flash "@flash_args"
