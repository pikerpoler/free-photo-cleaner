import Foundation
import Photos

@objc(StorageInfoModule)
class StorageInfoModule: NSObject {

  @objc
  static func requiresMainQueueSetup() -> Bool {
    return false
  }

  @objc
  func getStorageInfo(_ resolve: @escaping RCTPromiseResolveBlock, rejecter reject: @escaping RCTPromiseRejectBlock) {
    DispatchQueue.global(qos: .userInitiated).async {
      var totalSpace: Int64 = 0
      var freeSpace: Int64 = 0

      let homeURL = URL(fileURLWithPath: NSHomeDirectory())
      do {
        let values = try homeURL.resourceValues(forKeys: [
          .volumeTotalCapacityKey,
          .volumeAvailableCapacityForImportantUsageKey,
        ])
        totalSpace = Int64(values.volumeTotalCapacity ?? 0)
        if let available = values.volumeAvailableCapacityForImportantUsage {
          freeSpace = available
        }
      } catch {
        NSLog("[FreePhotoCleaner] StorageInfo disk space error: %@", error.localizedDescription)
        if let attrs = try? FileManager.default.attributesOfFileSystem(forPath: NSHomeDirectory()) {
          totalSpace = (attrs[.systemSize] as? Int64) ?? 0
          freeSpace = (attrs[.systemFreeSize] as? Int64) ?? 0
        }
      }

      NSLog("[FreePhotoCleaner] StorageInfo: totalSpace=%lld, freeSpace=%lld", totalSpace, freeSpace)

      let fetchOptions = PHFetchOptions()
      fetchOptions.includeHiddenAssets = false
      fetchOptions.includeAllBurstAssets = false

      let photoAssets = PHAsset.fetchAssets(with: .image, options: fetchOptions)
      let videoAssets = PHAsset.fetchAssets(with: .video, options: fetchOptions)

      let photoCount = photoAssets.count
      let videoCount = videoAssets.count

      NSLog("[FreePhotoCleaner] StorageInfo: photoCount=%d, videoCount=%d", photoCount, videoCount)

      let photosSize = self.estimateTotalSize(fetchResult: photoAssets, sampleCount: 50)
      let videosSize = self.estimateTotalSize(fetchResult: videoAssets, sampleCount: 30)

      NSLog("[FreePhotoCleaner] StorageInfo: photosSize=%lld, videosSize=%lld", photosSize, videosSize)

      let result: [String: Any] = [
        "totalSpace": NSNumber(value: totalSpace),
        "freeSpace": NSNumber(value: freeSpace),
        "photosSize": NSNumber(value: photosSize),
        "videosSize": NSNumber(value: videosSize),
        "photoCount": NSNumber(value: photoCount),
        "videoCount": NSNumber(value: videoCount),
      ]

      resolve(result)
    }
  }

  private func estimateTotalSize(fetchResult: PHFetchResult<PHAsset>, sampleCount: Int) -> Int64 {
    let totalCount = fetchResult.count
    guard totalCount > 0 else { return 0 }

    let samplesToTake = min(sampleCount, totalCount)
    var sampleSizeSum: Int64 = 0
    var samplesCollected = 0

    let stride = max(1, totalCount / samplesToTake)

    for i in Swift.stride(from: 0, to: totalCount, by: stride) {
      if samplesCollected >= samplesToTake { break }
      let asset = fetchResult.object(at: i)
      if let size = self.getAssetFileSize(asset) {
        sampleSizeSum += size
        samplesCollected += 1
      }
    }

    guard samplesCollected > 0 else { return 0 }

    let averageSize = sampleSizeSum / Int64(samplesCollected)
    return averageSize * Int64(totalCount)
  }

  private func getAssetFileSize(_ asset: PHAsset) -> Int64? {
    let resources = PHAssetResource.assetResources(for: asset)
    guard let primaryResource = resources.first else { return nil }

    if let size = primaryResource.value(forKey: "fileSize") as? Int64, size > 0 {
      return size
    }
    if let size = primaryResource.value(forKey: "fileSize") as? Int, size > 0 {
      return Int64(size)
    }

    // Fallback: use estimated file size from asset dimensions and type
    if asset.mediaType == .image {
      let pixels = Int64(asset.pixelWidth) * Int64(asset.pixelHeight)
      // ~3 bytes per pixel for compressed HEIC/JPEG is a reasonable estimate
      return max(pixels * 3 / 4, 500_000)
    } else if asset.mediaType == .video {
      let duration = Int64(asset.duration)
      // ~5MB per minute for typical video
      return max(duration * 85_000, 1_000_000)
    }

    return nil
  }

  @objc
  func batchDelete(_ uris: [String], resolve: @escaping RCTPromiseResolveBlock, rejecter reject: @escaping RCTPromiseRejectBlock) {
    NSLog("[FreePhotoCleaner] batchDelete called with %d URIs", uris.count)

    guard !uris.isEmpty else {
      resolve(["success": true, "deletedCount": 0])
      return
    }

    var localIdentifiers: [String] = []

    for uri in uris {
      if uri.hasPrefix("ph://") {
        let identifier = String(uri.dropFirst(5))
        localIdentifiers.append(identifier)
      } else {
        localIdentifiers.append(uri)
      }
    }

    NSLog("[FreePhotoCleaner] Converted to %d local identifiers", localIdentifiers.count)
    if let first = localIdentifiers.first {
      NSLog("[FreePhotoCleaner] First identifier: %@", first)
    }

    let fetchResult = PHAsset.fetchAssets(withLocalIdentifiers: localIdentifiers, options: nil)
    NSLog("[FreePhotoCleaner] PHAsset.fetchAssets returned %d assets", fetchResult.count)

    if fetchResult.count == 0 {
      NSLog("[FreePhotoCleaner] No assets found, trying UUID-only approach")
      var uuidIdentifiers: [String] = []
      for id in localIdentifiers {
        let components = id.components(separatedBy: "/")
        if components.count > 0 {
          uuidIdentifiers.append(components[0])
        }
      }

      let retryResult = PHAsset.fetchAssets(withLocalIdentifiers: uuidIdentifiers, options: nil)
      NSLog("[FreePhotoCleaner] UUID-only fetch returned %d assets", retryResult.count)

      if retryResult.count == 0 {
        NSLog("[FreePhotoCleaner] Still no assets found. Trying prefix match...")
        let allAssets = PHAsset.fetchAssets(with: nil)
        var matchedAssets: [PHAsset] = []
        let uuidSet = Set(uuidIdentifiers)

        allAssets.enumerateObjects { (asset, _, stop) in
          let assetUUID = asset.localIdentifier.components(separatedBy: "/").first ?? ""
          if uuidSet.contains(assetUUID) {
            matchedAssets.append(asset)
          }
          if matchedAssets.count >= uris.count {
            stop.pointee = true
          }
        }

        NSLog("[FreePhotoCleaner] Prefix match found %d assets", matchedAssets.count)

        if matchedAssets.isEmpty {
          resolve(["success": false, "deletedCount": 0, "error": "No matching assets found in photo library"])
          return
        }

        self.performDeletion(assets: matchedAssets, resolve: resolve)
        return
      }

      self.performDeletionFromFetchResult(retryResult, resolve: resolve)
      return
    }

    self.performDeletionFromFetchResult(fetchResult, resolve: resolve)
  }

  private func performDeletionFromFetchResult(_ fetchResult: PHFetchResult<PHAsset>, resolve: @escaping RCTPromiseResolveBlock) {
    let count = fetchResult.count
    NSLog("[FreePhotoCleaner] Requesting deletion of %d assets", count)

    PHPhotoLibrary.shared().performChanges({
      PHAssetChangeRequest.deleteAssets(fetchResult)
    }) { success, error in
      if success {
        NSLog("[FreePhotoCleaner] Successfully deleted %d assets", count)
        resolve(["success": true, "deletedCount": count])
      } else {
        let errorMsg = error?.localizedDescription ?? "User denied or unknown error"
        NSLog("[FreePhotoCleaner] Delete failed: %@", errorMsg)
        resolve(["success": false, "deletedCount": 0, "error": errorMsg])
      }
    }
  }

  private func performDeletion(assets: [PHAsset], resolve: @escaping RCTPromiseResolveBlock) {
    let count = assets.count
    NSLog("[FreePhotoCleaner] Requesting deletion of %d matched assets", count)

    PHPhotoLibrary.shared().performChanges({
      PHAssetChangeRequest.deleteAssets(assets as NSFastEnumeration)
    }) { success, error in
      if success {
        NSLog("[FreePhotoCleaner] Successfully deleted %d assets", count)
        resolve(["success": true, "deletedCount": count])
      } else {
        let errorMsg = error?.localizedDescription ?? "User denied or unknown error"
        NSLog("[FreePhotoCleaner] Delete failed: %@", errorMsg)
        resolve(["success": false, "deletedCount": 0, "error": errorMsg])
      }
    }
  }
}
