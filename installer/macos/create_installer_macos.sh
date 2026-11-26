##!/bin/bash

conda activate papylio
pyinstaller pyinstaller_configuration_macos.spec --noconfirm
dmgbuild -s dmgbuild_configuration.py "Papylio" ../../dist/papylio-installer.dmg
conda deactivate
#set -e
#
## Configuration
#APP_NAME="Papylio"
#VOLUME_NAME="Papylio"
#DIST_DIR="dist/mac"
#DMG_DIR="dist/dmg"
#DMG_PATH="dist/Papylio.dmg"
#
#APP_PATH="$DIST_DIR/$APP_NAME.app"
#
#echo "Creating DMG for $APP_NAME..."
#
## Check that the app exists
#if [ ! -d "$APP_PATH" ]; then
#    echo "Error: $APP_PATH not found."
#    echo "Build the application with PyInstaller first."
#    exit 1
#fi
#
## Clean previous DMG build
##rm -rf "$DMG_DIR"
#rm -f "$DMG_PATH"
#
## Create temporary DMG contents
#mkdir -p "$DMG_DIR"
#
#echo "Copying application..."
#cp -R "$APP_PATH" "$DMG_DIR/"
#
## Create Applications shortcut
#ln -s /Applications "$DMG_DIR/Applications"
#
## Create compressed DMG
#echo "Creating DMG..."
#hdiutil create \
#    -volname "$VOLUME_NAME" \
#    -srcfolder "$DMG_DIR" \
#    -ov \
#    -format UDZO \
#    "$DMG_PATH"
#
## Clean up temporary files
#rm -rf "$DMG_DIR"
#
#echo ""
#echo "DMG created successfully:"
#echo "$DMG_PATH"