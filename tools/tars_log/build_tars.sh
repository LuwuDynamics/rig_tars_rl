#!/bin/zsh
# Build the Tars firmware (国内版) into build/packages/tars-cn - the same recipe tools/package_luwu_firmware.py uses.
# usage: tools/tars_log/build_tars.sh            (needs ESP-IDF 5.5.x; IDF_PATH defaults to ~/esp/esp-idf-v5.5.3)
set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export IDF_PATH="${IDF_PATH:-$HOME/esp/esp-idf-v5.5.3}"
source "$IDF_PATH/export.sh" > /dev/null 2>&1 || true
# the xtensa compiler is not always on PATH after export.sh: add the one ESP-IDF 5.5.x expects (esp-14.2.0_20251107; a newer 15.x is NOT accepted by its build), else the newest installed
TC="$(ls -d $HOME/.espressif/tools/xtensa-esp-elf/esp-14.2.0_20251107/xtensa-esp-elf/bin 2>/dev/null | tail -1)"
[ -z "$TC" ] && TC="$(ls -d $HOME/.espressif/tools/xtensa-esp-elf/*/xtensa-esp-elf/bin 2>/dev/null | tail -1)"
[ -n "$TC" ] && export PATH="$TC:$PATH"
cd "$ROOT"
mkdir -p build/package-configs
cat > build/package-configs/sdkconfig.tars-cn <<'CFG'
CONFIG_IDF_TARGET="esp32s3"
CONFIG_IDF_TARGET_ESP32S3=y

# CONFIG_BOARD_TYPE_PUPPY is not set
# CONFIG_BOARD_TYPE_HOVER is not set
# CONFIG_BOARD_TYPE_ARM is not set
CONFIG_BOARD_TYPE_TARS=y

CONFIG_FIRMWARE_REGION_DOMESTIC=y
# CONFIG_FIRMWARE_REGION_OVERSEAS is not set
# CONFIG_SR_WN_WN9_HEYKIRA_TTS3 is not set
CONFIG_FLASH_EXPRESSION_ASSETS=y
CONFIG_USE_EMOTE_MESSAGE_STYLE=y
CFG
idf.py -B build/packages/tars-cn \
  -DSDKCONFIG="$ROOT/build/package-configs/sdkconfig.tars-cn" \
  "-DSDKCONFIG_DEFAULTS=$ROOT/sdkconfig.defaults;$ROOT/sdkconfig.defaults.esp32s3" \
  -DIDF_TARGET=esp32s3 build
echo
echo "built: $ROOT/build/packages/tars-cn/rig-tars.bin   -> flash with: tools/tars_log/flash_tars.sh /dev/cu.usbmodemXXXX"
