#!/bin/bash

set -e

INPUT="icon.png"
ICONSET="icon.iconset"
OUTPUT="icon.icns"

rm -rf "$ICONSET"
mkdir -p "$ICONSET"

sips -z 16 16       "$INPUT" --out "$ICONSET/icon_16x16.png"
sips -z 32 32       "$INPUT" --out "$ICONSET/icon_16x16@2x.png"
sips -z 32 32       "$INPUT" --out "$ICONSET/icon_32x32.png"
sips -z 64 64       "$INPUT" --out "$ICONSET/icon_32x32@2x.png"
sips -z 128 128     "$INPUT" --out "$ICONSET/icon_128x128.png"
sips -z 256 256     "$INPUT" --out "$ICONSET/icon_128x128@2x.png"
sips -z 256 256     "$INPUT" --out "$ICONSET/icon_256x256.png"
sips -z 512 512     "$INPUT" --out "$ICONSET/icon_256x256@2x.png"
sips -z 512 512     "$INPUT" --out "$ICONSET/icon_512x512.png"
sips -z 1024 1024   "$INPUT" --out "$ICONSET/icon_512x512@2x.png"

iconutil -c icns "$ICONSET" -o "$OUTPUT"

rm -rf "$ICONSET"

echo "Created $OUTPUT"