#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BUILD_TYPE="${1:-release}"
OUTPUT_DIR="$PROJECT_ROOT/build-output/android"

echo "========================================"
echo "  FreePhotoCleaner Android Build"
echo "  Type: $BUILD_TYPE"
echo "========================================"
echo ""

# ─── Prerequisites ───────────────────────────────────────────────────────────

if [ "$BUILD_TYPE" = "help" ]; then
    echo "Usage: $0 [debug|release|help]"
    echo ""
    echo "  debug     Build debug APK (for development)"
    echo "  release   Build release APK (for distribution)"
    echo ""
    echo "Requirements:"
    echo "  - Java JDK 17+"
    echo "  - Android SDK (via ANDROID_HOME or ANDROID_SDK_ROOT)"
    echo ""
    echo "Optional environment variables for release signing:"
    echo "  ANDROID_KEYSTORE_PATH    Path to .keystore file"
    echo "  ANDROID_KEYSTORE_PASS    Keystore password"
    echo "  ANDROID_KEY_ALIAS        Key alias"
    echo "  ANDROID_KEY_PASS         Key password"
    echo ""
    echo "Examples:"
    echo "  $0 debug      # Debug APK for testing"
    echo "  $0 release    # Release APK"
    exit 0
fi

if ! command -v java &>/dev/null; then
    echo "ERROR: Java not found. Install JDK 17+ to build Android."
    echo "  brew install openjdk@17"
    exit 1
fi

JAVA_VERSION=$(java -version 2>&1 | head -1 | cut -d'"' -f2 | cut -d'.' -f1)
if [ "$JAVA_VERSION" -lt 17 ] 2>/dev/null; then
    echo "WARNING: JDK 17+ recommended. Found version $JAVA_VERSION."
fi

mkdir -p "$OUTPUT_DIR"

cd "$PROJECT_ROOT/android"

if [ "$BUILD_TYPE" = "release" ]; then
    echo "[1/3] Cleaning previous build..."
    ./gradlew clean 2>/dev/null || true

    echo "[2/3] Building release APK..."
    ./gradlew assembleRelease

    APK_PATH="app/build/outputs/apk/release/app-release.apk"
    if [ -f "$APK_PATH" ]; then
        cp "$APK_PATH" "$OUTPUT_DIR/FreePhotoCleaner-release.apk"
        echo ""
        echo "[3/3] Build complete!"
        echo "  APK: $OUTPUT_DIR/FreePhotoCleaner-release.apk"
        echo "  Size: $(du -h "$OUTPUT_DIR/FreePhotoCleaner-release.apk" | cut -f1)"
    else
        echo "ERROR: APK not found at expected path."
        echo "Searching for APK files..."
        find app/build/outputs -name "*.apk" 2>/dev/null || echo "  No APKs found."
        exit 1
    fi

elif [ "$BUILD_TYPE" = "debug" ]; then
    echo "[1/3] Cleaning previous build..."
    ./gradlew clean 2>/dev/null || true

    echo "[2/3] Building debug APK..."
    ./gradlew assembleDebug

    APK_PATH="app/build/outputs/apk/debug/app-debug.apk"
    if [ -f "$APK_PATH" ]; then
        cp "$APK_PATH" "$OUTPUT_DIR/FreePhotoCleaner-debug.apk"
        echo ""
        echo "[3/3] Build complete!"
        echo "  APK: $OUTPUT_DIR/FreePhotoCleaner-debug.apk"
        echo "  Size: $(du -h "$OUTPUT_DIR/FreePhotoCleaner-debug.apk" | cut -f1)"
    else
        echo "ERROR: APK not found."
        exit 1
    fi

else
    echo "Usage: $0 [debug|release|help]"
    exit 1
fi

echo ""
echo "To install on a connected device:"
echo "  adb install -r $OUTPUT_DIR/FreePhotoCleaner-${BUILD_TYPE}.apk"
echo ""
