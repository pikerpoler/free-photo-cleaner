# Build & Deploy

## Prerequisites

| Requirement        | Version          |
|--------------------|------------------|
| Node.js            | >= 22.11.0       |
| Xcode              | >= 16.1 (iOS)    |
| JDK                | 17+ (Android)    |
| Android NDK        | 27.1.12297006    |
| CocoaPods          | via Bundler      |

## Development

```sh
npm install
cd ios && bundle exec pod install && cd ..

npm start          # Metro bundler
npm run ios        # iOS simulator
npm run android    # Android emulator/device
```

## Production Builds

### iOS

| Command                    | Output                           |
|----------------------------|----------------------------------|
| `npm run build:ios`        | `build-output/ios/FreePhotoCleaner.ipa` (release, signed) |
| `npm run build:ios:simulator` | `.app` in DerivedData (simulator) |

**Script**: `scripts/build-ios.sh`

The script:
1. Runs `pod install`
2. Creates a temp keychain (if signing env vars set)
3. `xcodebuild archive` → `build-output/ios/FreePhotoCleaner.xcarchive`
4. `xcodebuild -exportArchive` → `.ipa`
5. Cleans up temp keychain

**Signing env vars** (optional — uses auto-signing if unset):
- `IOS_TEAM_ID`
- `IOS_CERT_PATH`, `IOS_CERT_PASSWORD`
- `IOS_PROFILE_PATH`

**Bundle ID**: `com.freephotocleaner.app`

### Android

| Command                          | Output                                     |
|----------------------------------|--------------------------------------------|
| `npm run build:android`         | `build-output/android/FreePhotoCleaner-release.apk` |
| `npm run build:android -- debug` | `build-output/android/FreePhotoCleaner-debug.apk` |

**Script**: `scripts/build-android.sh`

Runs `./gradlew assembleRelease` or `assembleDebug`, copies APK to build-output.

**Signing env vars** (release only):
- `ANDROID_KEYSTORE_PATH`
- `ANDROID_KEYSTORE_PASSWORD`
- `ANDROID_KEY_ALIAS`
- `ANDROID_KEY_PASSWORD`

## Output Directory

```
build-output/
├── ios/
│   ├── FreePhotoCleaner.ipa
│   └── FreePhotoCleaner.xcarchive/
└── android/
    └── FreePhotoCleaner-release.apk
```

`build-output/` is in `.gitignore`.

## Other npm Scripts

| Script       | Purpose                    |
|--------------|----------------------------|
| `npm test`   | Jest tests                 |
| `npm run lint` | ESLint                   |
| `npm run pods` | `cd ios && pod install`  |
