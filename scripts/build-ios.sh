#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SCHEME="FreePhotoCleaner"
WORKSPACE=""
XCODEPROJ="$PROJECT_ROOT/ios/FreePhotoCleaner.xcodeproj"
OUTPUT_DIR="$PROJECT_ROOT/build-output/ios"
ARCHIVE_PATH="$OUTPUT_DIR/FreePhotoCleaner.xcarchive"
EXPORT_PATH="$OUTPUT_DIR/export"
BUNDLE_ID="com.freephotocleaner.app"
TEAM_ID="${IOS_TEAM_ID:-}"
KEYCHAIN_NAME="freephotocleaner-build.keychain"
KEYCHAIN_PASSWORD="fpc-temp-$(date +%s)"

# ─── Signing configurations ──────────────────────────────────────────────────
# All signing values come from environment variables or CLI flags.
# See '$0 help' for details, or create a .env file (gitignored) with:
#   export IOS_TEAM_ID="YOUR_TEAM_ID"
#   export IOS_CERT_PATH="/path/to/cert.p12"
#   export IOS_CERT_PASS="your-password"
#   export IOS_PROFILE_PATH="/path/to/profile.mobileprovision"
#   export IOS_SIGN_IDENTITY="iPhone Distribution: Your Name (TEAM_ID)"
#   export IOS_EXPORT_METHOD="ad-hoc"

CERT_PATH="${IOS_CERT_PATH:-}"
CERT_PASS="${IOS_CERT_PASS:-}"
PROFILE_PATH="${IOS_PROFILE_PATH:-}"
CODE_SIGN_IDENTITY="${IOS_SIGN_IDENTITY:-}"
EXPORT_METHOD="${IOS_EXPORT_METHOD:-ad-hoc}"

# ─── Parse arguments ─────────────────────────────────────────────────────────

BUILD_TYPE=""

for arg in "$@"; do
    case "$arg" in
        release|ipa|debug|simulator|help) BUILD_TYPE="$arg" ;;
        --team=*)   TEAM_ID="${arg#*=}" ;;
        --cert=*)   CERT_PATH="${arg#*=}" ;;
        --pass=*)   CERT_PASS="${arg#*=}" ;;
        --profile=*) PROFILE_PATH="${arg#*=}" ;;
        --identity=*) CODE_SIGN_IDENTITY="${arg#*=}" ;;
        --method=*)  EXPORT_METHOD="${arg#*=}" ;;
        *)
            echo "Unknown argument: $arg"
            exit 1
            ;;
    esac
done

BUILD_TYPE="${BUILD_TYPE:-release}"

echo "========================================"
echo "  FreePhotoCleaner iOS Build"
echo "  Type: $BUILD_TYPE"
echo "========================================"
echo ""

# ─── Usage ───────────────────────────────────────────────────────────────────

if [ "$BUILD_TYPE" = "help" ]; then
    echo "Usage: $0 [release|ipa|debug|simulator|help] [options]"
    echo ""
    echo "Build types:"
    echo "  release/ipa      Archive + export signed IPA"
    echo "  debug/simulator   Build for iOS Simulator (no signing needed)"
    echo ""
    echo "Signing options (for release builds):"
    echo "  --team=TEAM_ID        Apple Developer Team ID"
    echo "  --cert=PATH           Path to .p12 certificate"
    echo "  --pass=PASSWORD       Certificate password"
    echo "  --profile=PATH        Path to .mobileprovision"
    echo "  --identity=IDENTITY   Code sign identity"
    echo "  --method=METHOD       Export method (ad-hoc, app-store, development)"
    echo ""
    echo "Environment variables (alternative to flags):"
    echo "  IOS_TEAM_ID, IOS_CERT_PATH, IOS_CERT_PASS"
    echo "  IOS_PROFILE_PATH, IOS_SIGN_IDENTITY, IOS_EXPORT_METHOD"
    echo ""
    echo "Examples:"
    echo "  $0 simulator                     # Simulator build (no signing)"
    echo "  $0 release --team=ABC123         # Release with signing"
    echo "  $0 release --cert=cert.p12 --pass=secret --profile=app.mobileprovision"
    exit 0
fi

# ─── Prerequisites ───────────────────────────────────────────────────────────

if ! command -v xcodebuild &>/dev/null; then
    echo "ERROR: xcodebuild not found. Install Xcode."
    exit 1
fi

# Auto-detect Xcode if the active one is too old (RN 0.85+ requires Xcode >= 16.1)
XCODE_VERSION_OUTPUT=$(xcodebuild -version 2>/dev/null || true)
XCODE_MAJOR=$(echo "$XCODE_VERSION_OUTPUT" | grep -o 'Xcode [0-9]*' | grep -o '[0-9]*' || true)
if [ -n "$XCODE_MAJOR" ] && [ "$XCODE_MAJOR" -lt 16 ] 2>/dev/null; then
    if [ -d "/Applications/Xcode.app/Contents/Developer" ]; then
        echo "[*] Active Xcode ($XCODE_MAJOR.x) is too old. Using /Applications/Xcode.app instead."
        export DEVELOPER_DIR="/Applications/Xcode.app/Contents/Developer"
    else
        echo "WARNING: Active Xcode version $XCODE_MAJOR.x may be too old. Xcode 16.1+ is required."
    fi
fi

# Check for workspace (created after pod install)
if [ -f "$PROJECT_ROOT/ios/FreePhotoCleaner.xcworkspace/contents.xcworkspacedata" ]; then
    WORKSPACE="$PROJECT_ROOT/ios/FreePhotoCleaner.xcworkspace"
fi

# ─── CocoaPods ───────────────────────────────────────────────────────────────

install_pods() {
    echo "[*] Installing CocoaPods dependencies..."
    cd "$PROJECT_ROOT/ios"

    export LANG=en_US.UTF-8
    export LC_ALL=en_US.UTF-8

    local POD_CMD=""
    if command -v pod &>/dev/null; then
        POD_CMD="pod"
    elif [ -n "$(command -v /opt/homebrew/lib/ruby/gems/*/bin/pod 2>/dev/null | head -1)" ]; then
        POD_CMD="$(command -v /opt/homebrew/lib/ruby/gems/*/bin/pod 2>/dev/null | head -1)"
    elif command -v bundle &>/dev/null && [ -f "$PROJECT_ROOT/Gemfile" ]; then
        bundle install --quiet 2>/dev/null || true
        POD_CMD="bundle exec pod"
    else
        echo "ERROR: CocoaPods not found."
        echo "  Install with: sudo gem install cocoapods"
        echo "  Or: brew install cocoapods"
        exit 1
    fi

    $POD_CMD install
    WORKSPACE="$PROJECT_ROOT/ios/FreePhotoCleaner.xcworkspace"
    cd "$PROJECT_ROOT"
}

build_project_flag() {
    if [ -n "$WORKSPACE" ]; then
        echo "-workspace $WORKSPACE"
    else
        echo "-project $XCODEPROJ"
    fi
}

# ─── Keychain + Signing Setup ────────────────────────────────────────────────

setup_signing() {
    if [ -z "$CERT_PATH" ] || [ -z "$PROFILE_PATH" ]; then
        echo "ERROR: Signing requires --cert and --profile (or env vars IOS_CERT_PATH, IOS_PROFILE_PATH)."
        echo "  Run '$0 help' for usage."
        exit 1
    fi

    if [ -z "$TEAM_ID" ]; then
        echo "ERROR: Team ID required. Use --team=TEAM_ID or set IOS_TEAM_ID."
        exit 1
    fi

    if [ ! -f "$CERT_PATH" ]; then
        echo "ERROR: Certificate not found: $CERT_PATH"
        exit 1
    fi
    if [ ! -f "$PROFILE_PATH" ]; then
        echo "ERROR: Provisioning profile not found: $PROFILE_PATH"
        exit 1
    fi

    echo "[*] Setting up signing..."
    echo "  Identity: $CODE_SIGN_IDENTITY"
    echo "  Team: $TEAM_ID"
    echo "  Method: $EXPORT_METHOD"
    echo ""

    # Create a temporary keychain
    security create-keychain -p "$KEYCHAIN_PASSWORD" "$KEYCHAIN_NAME" 2>/dev/null || true
    security set-keychain-settings -lut 3600 "$KEYCHAIN_NAME"
    security unlock-keychain -p "$KEYCHAIN_PASSWORD" "$KEYCHAIN_NAME"

    # Add temp keychain to the search list (preserve existing)
    local EXISTING_KEYCHAINS
    EXISTING_KEYCHAINS=$(security list-keychains -d user | tr -d '"' | tr '\n' ' ')
    security list-keychains -d user -s "$KEYCHAIN_NAME" $EXISTING_KEYCHAINS

    # Import the certificate
    security import "$CERT_PATH" \
        -k "$KEYCHAIN_NAME" \
        -P "$CERT_PASS" \
        -T /usr/bin/codesign \
        -T /usr/bin/security

    # Allow codesign to access the keychain without prompt
    security set-key-partition-list -S apple-tool:,apple:,codesign: \
        -s -k "$KEYCHAIN_PASSWORD" "$KEYCHAIN_NAME" >/dev/null 2>&1 || true

    # Install provisioning profile
    local PROFILE_UUID
    PROFILE_UUID=$(security cms -D -i "$PROFILE_PATH" 2>/dev/null \
        | plutil -extract UUID raw -o - - 2>/dev/null)

    mkdir -p "$HOME/Library/MobileDevice/Provisioning Profiles"
    cp "$PROFILE_PATH" "$HOME/Library/MobileDevice/Provisioning Profiles/${PROFILE_UUID}.mobileprovision"

    echo "  Profile UUID: $PROFILE_UUID"
    echo ""

    PROVISIONING_PROFILE_UUID="$PROFILE_UUID"
}

cleanup_signing() {
    echo "[*] Cleaning up temporary keychain..."
    security delete-keychain "$KEYCHAIN_NAME" 2>/dev/null || true
}

# ─── Build ───────────────────────────────────────────────────────────────────

mkdir -p "$OUTPUT_DIR"

if [ "$BUILD_TYPE" = "release" ] || [ "$BUILD_TYPE" = "ipa" ]; then
    if [ -z "$WORKSPACE" ]; then
        install_pods
    fi

    setup_signing
    trap cleanup_signing EXIT

    echo "[1/4] Cleaning previous build..."
    xcodebuild clean \
        $(build_project_flag) \
        -scheme "$SCHEME" \
        -configuration Release \
        -quiet 2>/dev/null || true

    echo "[2/4] Archiving..."
    xcodebuild archive \
        $(build_project_flag) \
        -scheme "$SCHEME" \
        -configuration Release \
        -destination "generic/platform=iOS" \
        -archivePath "$ARCHIVE_PATH" \
        CODE_SIGN_STYLE="Manual" \
        DEVELOPMENT_TEAM="$TEAM_ID" \
        CODE_SIGN_IDENTITY="$CODE_SIGN_IDENTITY" \
        PROVISIONING_PROFILE="$PROVISIONING_PROFILE_UUID" \
        PRODUCT_BUNDLE_IDENTIFIER="$BUNDLE_ID" \
        OTHER_CODE_SIGN_FLAGS="--keychain $KEYCHAIN_NAME"

    if [ ! -d "$ARCHIVE_PATH" ]; then
        echo "ERROR: Archive failed. Run xcodebuild directly for full errors."
        exit 1
    fi

    echo "[3/4] Exporting IPA..."

    EXPORT_PLIST="$OUTPUT_DIR/ExportOptions.plist"
    cat > "$EXPORT_PLIST" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>method</key>
    <string>${EXPORT_METHOD}</string>
    <key>compileBitcode</key>
    <false/>
    <key>stripSwiftSymbols</key>
    <true/>
    <key>signingStyle</key>
    <string>manual</string>
    <key>teamID</key>
    <string>${TEAM_ID}</string>
    <key>signingCertificate</key>
    <string>${CODE_SIGN_IDENTITY}</string>
    <key>provisioningProfiles</key>
    <dict>
        <key>${BUNDLE_ID}</key>
        <string>${PROVISIONING_PROFILE_UUID}</string>
    </dict>
</dict>
</plist>
PLIST

    xcodebuild -exportArchive \
        -archivePath "$ARCHIVE_PATH" \
        -exportPath "$EXPORT_PATH" \
        -exportOptionsPlist "$EXPORT_PLIST"

    IPA_FILE=$(find "$EXPORT_PATH" -name "*.ipa" -print -quit 2>/dev/null)
    if [ -n "$IPA_FILE" ]; then
        cp "$IPA_FILE" "$OUTPUT_DIR/FreePhotoCleaner.ipa"
        echo ""
        echo "[4/4] Build complete!"
        echo "  IPA: $OUTPUT_DIR/FreePhotoCleaner.ipa"
        echo "  Archive: $ARCHIVE_PATH"
        echo "  Size: $(du -h "$OUTPUT_DIR/FreePhotoCleaner.ipa" | cut -f1)"
    else
        echo ""
        echo "[4/4] Archive created but IPA export failed."
        echo "  Archive: $ARCHIVE_PATH"
        echo "  You can export manually: open $ARCHIVE_PATH"
    fi

elif [ "$BUILD_TYPE" = "debug" ] || [ "$BUILD_TYPE" = "simulator" ]; then
    if [ -z "$WORKSPACE" ]; then
        install_pods
    fi

    echo "[1/2] Building for simulator..."
    xcodebuild build \
        $(build_project_flag) \
        -scheme "$SCHEME" \
        -configuration Debug \
        -destination "generic/platform=iOS Simulator" \
        -derivedDataPath "$OUTPUT_DIR/DerivedData"

    APP_PATH=$(find "$OUTPUT_DIR/DerivedData" -name "FreePhotoCleaner.app" -print -quit 2>/dev/null)
    if [ -n "$APP_PATH" ]; then
        echo ""
        echo "[2/2] Build complete!"
        echo "  App: $APP_PATH"
        echo ""
        echo "To install on simulator:"
        echo "  xcrun simctl install booted '$APP_PATH'"
        echo "  xcrun simctl launch booted $BUNDLE_ID"
    else
        echo "ERROR: Build output not found."
        exit 1
    fi

else
    echo "Usage: $0 [release|ipa|debug|simulator|help] [options]"
    echo "Run '$0 help' for full usage."
    exit 1
fi

echo ""
