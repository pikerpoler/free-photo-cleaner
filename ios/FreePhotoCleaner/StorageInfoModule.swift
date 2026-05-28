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
        if let attrs = try? FileManager.default.attributesOfFileSystem(forPath: NSHomeDirectory()) {
          totalSpace = (attrs[.systemSize] as? Int64) ?? 0
          freeSpace = (attrs[.systemFreeSize] as? Int64) ?? 0
        }
      }

      let fetchOptions = PHFetchOptions()
      fetchOptions.includeHiddenAssets = false
      fetchOptions.includeAllBurstAssets = false

      var photosSize: Int64 = 0
      var videosSize: Int64 = 0

      let photoAssets = PHAsset.fetchAssets(with: .image, options: fetchOptions)
      let videoAssets = PHAsset.fetchAssets(with: .video, options: fetchOptions)

      photoAssets.enumerateObjects { (asset, _, _) in
        let resources = PHAssetResource.assetResources(for: asset)
        for resource in resources {
          if let size = resource.value(forKey: "fileSize") as? Int64 {
            photosSize += size
            break
          }
        }
      }

      videoAssets.enumerateObjects { (asset, _, _) in
        let resources = PHAssetResource.assetResources(for: asset)
        for resource in resources {
          if let size = resource.value(forKey: "fileSize") as? Int64 {
            videosSize += size
            break
          }
        }
      }

      let result: [String: Any] = [
        "totalSpace": NSNumber(value: totalSpace),
        "freeSpace": NSNumber(value: freeSpace),
        "photosSize": NSNumber(value: photosSize),
        "videosSize": NSNumber(value: videosSize),
      ]

      resolve(result)
    }
  }

  @objc
  func batchDelete(_ uris: [String], resolve: @escaping RCTPromiseResolveBlock, rejecter reject: @escaping RCTPromiseRejectBlock) {
    NSLog("[FreePhotoCleaner] batchDelete called with %d URIs", uris.count)

    guard !uris.isEmpty else {
      resolve(["success": true, "deletedCount": 0])
      return
    }

    // Convert ph:// URIs to local identifiers
    // CameraRoll returns: "ph://ED7AC36B-A150-4C38-BB8C-B6D696F4F2ED/L0/001"
    // PHAsset localIdentifier is: "ED7AC36B-A150-4C38-BB8C-B6D696F4F2ED/L0/001"
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

    // Try fetching with full identifiers first
    let fetchResult = PHAsset.fetchAssets(withLocalIdentifiers: localIdentifiers, options: nil)
    NSLog("[FreePhotoCleaner] PHAsset.fetchAssets returned %d assets", fetchResult.count)

    if fetchResult.count == 0 {
      // Fallback: try with just the UUID portion (strip /L0/001 suffix)
      NSLog("[FreePhotoCleaner] No assets found, trying UUID-only approach")
      var uuidIdentifiers: [String] = []
      for id in localIdentifiers {
        // localIdentifier format might just be UUID without path components
        let components = id.components(separatedBy: "/")
        if components.count > 0 {
          uuidIdentifiers.append(components[0])
        }
      }

      let retryResult = PHAsset.fetchAssets(withLocalIdentifiers: uuidIdentifiers, options: nil)
      NSLog("[FreePhotoCleaner] UUID-only fetch returned %d assets", retryResult.count)

      if retryResult.count == 0 {
        // Final attempt: try fetching ALL photos and matching by localIdentifier prefix
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
