import Foundation
import Photos
import UIKit
import Vision
import CoreML
import Accelerate

enum PhotoFeatures {
  private static var photometricCount = 0
  private static let photometricLock = NSLock()

  // MARK: - Metadata helpers

  static func isScreenshotFilename(_ filename: String) -> Bool {
    let lower = filename.lowercased()
    if lower.contains("screenshot") { return true }
    let png = lower.hasSuffix(".png") && lower.contains("img_")
    if png {
      let base = (filename as NSString).deletingPathExtension
      let digits = base.replacingOccurrences(of: "img_", with: "", options: .caseInsensitive)
      return digits.allSatisfy(\.isNumber)
    }
    return false
  }

  static func isWhatsAppFilename(_ filename: String) -> Bool {
    let lower = filename.lowercased()
    if lower.contains("whatsapp") { return true }
    return lower.range(of: #"img-\d+-wa"#, options: .regularExpression) != nil
  }

  static func isScreenshot(asset: PHAsset, filename: String) -> Bool {
    if asset.mediaSubtypes.contains(.photoScreenshot) { return true }
    return isScreenshotFilename(filename)
  }

  // MARK: - Photometric (16)

  static func photometric(from image: UIImage) -> [Float] {
    let startedAt = CFAbsoluteTimeGetCurrent()
    photometricLock.lock()
    photometricCount += 1
    let invocation = photometricCount
    photometricLock.unlock()
    guard let cg = image.cgImage else {
      return [Float](repeating: 0, count: FeatureSchema.photometricCount)
    }
    let w = cg.width
    let h = cg.height
    guard w > 0, h > 0 else {
      return [Float](repeating: 0, count: FeatureSchema.photometricCount)
    }
    let bytesPerPixel = 4
    let bytesPerRow = bytesPerPixel * w
    let rasterStartedAt = CFAbsoluteTimeGetCurrent()
    var data = [UInt8](repeating: 0, count: h * bytesPerRow)
    guard let ctx = CGContext(
      data: &data,
      width: w,
      height: h,
      bitsPerComponent: 8,
      bytesPerRow: bytesPerRow,
      space: CGColorSpaceCreateDeviceRGB(),
      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else {
      return [Float](repeating: 0, count: FeatureSchema.photometricCount)
    }
    ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
    let rasterMs = (CFAbsoluteTimeGetCurrent() - rasterStartedAt) * 1000

    let (features, timings) = data.withUnsafeBufferPointer { buf in
      PhotometricCore.compute(rgba: buf.baseAddress!, width: w, height: h, bytesPerRow: bytesPerRow)
    }
    NSLog(
      "[ScoreDiag] photometric n=%d pixels=%d rasterMs=%.1f pixelPassMs=%.1f " +
        "spatialPassMs=%.1f madMs=%.1f totalMs=%.1f",
      invocation,
      w * h,
      rasterMs,
      timings.pixelPassMs,
      timings.spatialPassMs,
      timings.madMs,
      (CFAbsoluteTimeGetCurrent() - startedAt) * 1000
    )
    return features
  }

  // MARK: - YOLO aggregation (32 = 6 geom + 26 cat)

  static func aggregateDetections(_ boxes: [DetectionBox]) -> [Float] {
    var out = [Float](repeating: 0, count: FeatureSchema.geometryCount + 2 * FeatureSchema.categoryCount)
    let filtered = boxes.filter { $0.conf >= FeatureSchema.confThreshold }
      .sorted { $0.conf > $1.conf }
      .prefix(FeatureSchema.maxBoxes)
    let dets = Array(filtered)
    guard !dets.isEmpty else { return out }

    var areas = [Float]()
    var touch = [Bool]()
    for d in dets {
      let w = max(0, min(1, d.x2) - max(0, d.x1))
      let h = max(0, min(1, d.y2) - max(0, d.y1))
      areas.append(w * h)
      touch.append(d.x1 <= 0.02 || d.y1 <= 0.02 || d.x2 >= 0.98 || d.y2 >= 0.98)
    }
    let largest = areas.indices.max(by: { areas[$0] < areas[$1] }) ?? 0
    let d = dets[largest]
    let cx = (d.x1 + d.x2) * 0.5
    let cy = (d.y1 + d.y2) * 0.5
    let dist = hypot(cx - 0.5, cy - 0.5)
    out[0] = log1p(Float(dets.count))
    out[1] = dets.map(\.conf).max() ?? 0
    out[2] = areas[largest]
    out[3] = dist
    out[4] = Float(touch.filter { $0 }.count) / Float(dets.count)
    out[5] = min(areas.reduce(0, +), 1)

    for box in dets {
      guard box.classId >= 0 && box.classId < 80 else { continue }
      let cat = Int(FeatureSchema.cocoToCategory[box.classId])
      guard cat >= 0 else { continue }
      let base = FeatureSchema.geometryCount + 2 * cat
      out[base] = max(out[base], box.conf)
      out[base + 1] += box.conf
    }
    return out
  }

  // MARK: - Vision (6)

  static func visionFeatures(faces: [(rect: CGRect, conf: Float)], textRects: [CGRect]) -> [Float] {
    var out = [Float](repeating: 0, count: FeatureSchema.visionCount)
    let goodFaces = faces.filter { $0.conf >= FeatureSchema.faceConfThreshold }
    if !goodFaces.isEmpty {
      out[0] = log1p(Float(goodFaces.count))
      out[1] = goodFaces.map(\.conf).max() ?? 0
      let areas = goodFaces.map { Float($0.rect.width * $0.rect.height) }
      let largest = areas.indices.max(by: { areas[$0] < areas[$1] }) ?? 0
      out[2] = areas[largest]
      let r = goodFaces[largest].rect
      let cx = r.midX
      let cy = r.midY
      out[3] = Float(hypot(cx - 0.5, cy - 0.5))
    }
    if !textRects.isEmpty {
      out[4] = log1p(Float(textRects.count))
      out[5] = min(textRects.map { Float($0.width * $0.height) }.reduce(0, +), 1)
    }
    return out
  }

  static func metadataVector(screenshot: Bool, whatsapp: Bool, aspect: Float) -> [Float] {
    [
      screenshot ? 1 : 0,
      whatsapp ? 1 : 0,
      aspect,
    ]
  }

  static func assemble(photo: [Float], yolo: [Float], vision: [Float], meta: [Float]) -> [Float] {
    photo + yolo + vision + meta
  }

  // MARK: - Cache

  static var featureCacheDir: URL {
    let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    let dir = docs.appendingPathComponent("ai_feature_cache/v\(FeatureSchema.version)", isDirectory: true)
    try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    return dir
  }

  static func cacheKey(localId: String, modDate: TimeInterval) -> String {
    let safe = localId.replacingOccurrences(of: "/", with: "_")
    return "\(safe)_\(Int(modDate))"
  }

  static func loadCachedFeatures(key: String) -> [Float]? {
    let url = featureCacheDir.appendingPathComponent("\(key).bin")
    guard let data = try? Data(contentsOf: url), data.count == FeatureSchema.dim * 4 else { return nil }
    return data.withUnsafeBytes { raw in
      Array(UnsafeBufferPointer(start: raw.bindMemory(to: Float.self).baseAddress, count: FeatureSchema.dim))
    }
  }

  static func saveCachedFeatures(key: String, vector: [Float]) {
    guard vector.count == FeatureSchema.dim else { return }
    var v = vector
    let data = Data(bytes: &v, count: v.count * 4)
    try? data.write(to: featureCacheDir.appendingPathComponent("\(key).bin"))
  }

  /// Remove cache files for one asset (all modificationDate variants).
  static func deleteCachedFeatures(localId: String) {
    let fm = FileManager.default
    let safe = localId.replacingOccurrences(of: "/", with: "_")
    let dir = featureCacheDir
    guard let urls = try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil) else { return }
    let prefix = safe + "_"
    for fileURL in urls {
      let name = fileURL.lastPathComponent
      if name.hasPrefix(prefix), name.hasSuffix(".bin") {
        try? fm.removeItem(at: fileURL)
      }
    }
  }

  /// `ph://` URIs from RN — removes cached `.bin` for each photo.
  static func deleteCachedFeatures(photoUris: [String]) {
    var targetSafe = Set<String>()
    targetSafe.reserveCapacity(photoUris.count)
    for uri in photoUris {
      guard let lid = localIdentifier(from: uri) else { continue }
      targetSafe.insert(lid.replacingOccurrences(of: "/", with: "_"))
    }
    guard !targetSafe.isEmpty else { return }
    let fm = FileManager.default
    let dir = featureCacheDir
    guard let urls = try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil) else { return }
    for fileURL in urls {
      let name = fileURL.lastPathComponent
      guard name.hasSuffix(".bin") else { continue }
      let base = (name as NSString).deletingPathExtension
      guard let u = base.lastIndex(of: "_") else { continue }
      let idPart = String(base[..<u])
      if targetSafe.contains(idPart) {
        try? fm.removeItem(at: fileURL)
      }
    }
  }

  /// After library sync: keep only caches whose `localId` is still in the catalog.
  static func pruneFeatureCache(livePhotoIds: Set<String>) {
    let fm = FileManager.default
    let dir = featureCacheDir
    guard let urls = try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil) else { return }
    var liveSafe = Set<String>()
    liveSafe.reserveCapacity(livePhotoIds.count)
    for id in livePhotoIds {
      liveSafe.insert(id.replacingOccurrences(of: "/", with: "_"))
    }
    for fileURL in urls {
      let name = fileURL.lastPathComponent
      guard name.hasSuffix(".bin") else { continue }
      let base = (name as NSString).deletingPathExtension
      guard let u = base.lastIndex(of: "_") else { continue }
      let idPart = String(base[..<u])
      if !liveSafe.contains(idPart) {
        try? fm.removeItem(at: fileURL)
      }
    }
  }

  /// Same as `pruneFeatureCache` but accepts `ph://…` URIs from JS (photo library ids).
  static func pruneFeatureCache(photoUris: [String]) {
    var ids = Set<String>()
    ids.reserveCapacity(photoUris.count)
    for uri in photoUris {
      if let lid = localIdentifier(from: uri) {
        ids.insert(lid)
      }
    }
    pruneFeatureCache(livePhotoIds: ids)
  }

  /// Drop `ai_feature_cache/v1` and `ai_feature_models/v1` after schema bump (v2 incompatible with zero-YOLO vectors).
  static func purgeLegacyFeatureCacheVersions(keeping: Int) {
    let fm = FileManager.default
    let docs = fm.urls(for: .documentDirectory, in: .userDomainMask)[0]
    for i in 1..<keeping {
      for root in ["ai_feature_cache", "ai_feature_models"] {
        let url = docs.appendingPathComponent("\(root)/v\(i)", isDirectory: true)
        if fm.fileExists(atPath: url.path) {
          try? fm.removeItem(at: url)
        }
      }
    }
  }

  // MARK: - Decode image

  static func decodeImage(asset: PHAsset, longEdge: Int) -> UIImage? {
    let options = PHImageRequestOptions()
    options.isSynchronous = true
    options.deliveryMode = .highQualityFormat
    options.resizeMode = .fast
    options.isNetworkAccessAllowed = false
    let w = CGFloat(asset.pixelWidth)
    let h = CGFloat(asset.pixelHeight)
    let scale = CGFloat(longEdge) / max(w, h, 1)
    let target = CGSize(width: max(1, w * scale), height: max(1, h * scale))
    var result: UIImage?
    PHImageManager.default().requestImage(
      for: asset,
      targetSize: target,
      contentMode: .aspectFit,
      options: options
    ) { img, _ in result = img }
    return result
  }

  static func resizeLongest(_ image: UIImage, longest: Int) -> UIImage? {
    let w = image.size.width * image.scale
    let h = image.size.height * image.scale
    let scale = CGFloat(longest) / max(w, h, 1)
    if scale >= 1 { return image }
    let nw = max(1, w * scale)
    let nh = max(1, h * scale)
    UIGraphicsBeginImageContextWithOptions(CGSize(width: nw, height: nh), true, 1)
    image.draw(in: CGRect(x: 0, y: 0, width: nw, height: nh))
    let out = UIGraphicsGetImageFromCurrentImageContext()
    UIGraphicsEndImageContext()
    return out
  }

  static func localIdentifier(from uri: String) -> String? {
    if uri.hasPrefix("ph://") {
      return String(uri.dropFirst(5))
    }
    if let url = URL(string: uri), url.scheme == "ph" {
      return url.host
    }
    return uri.contains("/") ? nil : uri
  }

  static func filename(for asset: PHAsset) -> String {
    let resources = PHAssetResource.assetResources(for: asset)
    return resources.first?.originalFilename ?? "unknown"
  }
}
