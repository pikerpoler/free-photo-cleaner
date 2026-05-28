# FreePhotoCleaner

A swipe-based photo and video cleanup app built with React Native. Swipe to keep or delete media, with undo support, filters, and offline-first storage.

## Prerequisites

- **Node.js** >= 22.11.0
- **Xcode** >= 16.1 (for iOS)
- **CocoaPods** — `brew install cocoapods` or `sudo gem install cocoapods`
- **Android SDK** with JDK 17+ (for Android)

## Setup

```sh
# Install JS dependencies
npm install

# Install iOS native dependencies
cd ios && pod install && cd ..
```

## Development

Start the Metro bundler, then run the app:

```sh
npm start
```

In a separate terminal:

```sh
# iOS (simulator)
npm run ios

# Android (emulator or device)
npm run android
```

You can also open the project directly in Xcode or Android Studio (see below).

## Opening in Xcode (without build scripts)

If you don't have the signing certificates referenced by the build scripts, you can still build and run via Xcode with automatic signing:

```sh
npm install
cd ios && pod install && cd ..
open ios/FreePhotoCleaner.xcworkspace
```

Then in Xcode:

1. Select the **FreePhotoCleaner** target
2. Go to **Signing & Capabilities**
3. Check **Automatically manage signing**
4. Select your **Team** (even a free Apple ID works for simulator/personal device)
5. Change the **Bundle Identifier** to something unique (e.g. `com.yourname.freephotocleaner`)
6. Select a simulator or connected device and press **Cmd+R** to run

## Production Builds

### iOS

The build script reads signing configuration from environment variables or CLI flags — no secrets are stored in the repo.

```sh
# Simulator build (no signing needed)
npm run build:ios:simulator

# Signed release IPA
export IOS_TEAM_ID="YOUR_TEAM_ID"
export IOS_CERT_PATH="/path/to/cert.p12"
export IOS_CERT_PASS="your-password"
export IOS_PROFILE_PATH="/path/to/profile.mobileprovision"
export IOS_SIGN_IDENTITY="iPhone Distribution: Your Name (TEAM_ID)"
export IOS_EXPORT_METHOD="ad-hoc"
npm run build:ios
```

Run `./scripts/build-ios.sh help` for all options.

### Android

```sh
# Debug APK
npm run build:android -- debug

# Release APK
npm run build:android
```

Run `./scripts/build-android.sh help` for all options.

Build outputs go to `build-output/`.

## Cleaning Build Artifacts

To reset the repo to a fresh-clone state:

```sh
# Remove JS dependencies
rm -rf node_modules

# Remove iOS native dependencies and build cache
rm -rf ios/Pods ios/build

# Remove Android build cache
rm -rf android/app/build android/app/.cxx android/.gradle

# Remove build outputs
rm -rf build-output

# Remove Ruby vendor bundle (if installed)
rm -rf vendor/bundle
```

Then re-run setup:

```sh
npm install
cd ios && pod install && cd ..
```

## Project Structure

```
src/
  components/    # Reusable UI components (SwipeCard, MediaCard, etc.)
  screens/       # App screens (Photos, Videos, Settings)
  navigation/    # React Navigation setup
  services/      # Database, media access, storage helpers
  stores/        # Zustand state stores
  types/         # TypeScript type definitions
  utils/         # Formatting utilities
scripts/
  build-ios.sh       # iOS archive/IPA build script
  build-android.sh   # Android APK build script
ios/                 # Xcode project and native iOS code
android/             # Android Studio project and native code
```
