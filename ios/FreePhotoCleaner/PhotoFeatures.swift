import Foundation
import Photos
import UIKit
import Vision
import CoreML
import Accelerate

enum PhotoFeatures {

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

    var lum = [Float]()
    lum.reserveCapacity(w * h)
    var sat = [Float]()
    sat.reserveCapacity(w * h)
    var reds = [Float]()
    var blues = [Float]()
    var rgDiff = [Float]()
    var ybDiff = [Float]()

    for y in 0..<h {
      for x in 0..<w {
        let i = y * bytesPerRow + x * 4
        let r = Float(data[i]) / 255
        let g = Float(data[i + 1]) / 255
        let b = Float(data[i + 2]) / 255
        let l = 0.2126 * r + 0.7152 * g + 0.0722 * b
        lum.append(l)
        reds.append(r)
        blues.append(b)
        let mx = max(r, max(g, b))
        let mn = min(r, min(g, b))
        let s = mx > 1e-6 ? (mx - mn) / mx : 0
        sat.append(s)
        rgDiff.append(r - g)
        ybDiff.append(0.5 * (r + g) - b)
      }
    }

    let n = Float(lum.count)
    let meanL = lum.reduce(0, +) / n
    var varL: Float = 0
    for v in lum { varL += (v - meanL) * (v - meanL) }
    let stdL = sqrt(varL / n)
    let fracDark = lum.filter { $0 < 0.05 }.count
    let fracBright = lum.filter { $0 > 0.95 }.count

    let lapVar = laplacianVariance(lum: lum, w: w, h: h)
    let (centerVar, borderVar) = centerBorderLaplacian(lum: lum, w: w, h: h)
    let ratio = centerVar / (borderVar + 1e-8)

    let meanSat = sat.reduce(0, +) / n
    var varSat: Float = 0
    for v in sat { varSat += (v - meanSat) * (v - meanSat) }
    let stdSat = sqrt(varSat / n)

    let rgMean = rgDiff.reduce(0, +) / n
    let ybMean = ybDiff.reduce(0, +) / n
    let rgStd = sqrt(rgDiff.map { ($0 - rgMean) * ($0 - rgMean) }.reduce(0, +) / n)
    let ybStd = sqrt(ybDiff.map { ($0 - ybMean) * ($0 - ybMean) }.reduce(0, +) / n)
    let colorfulness = sqrt(rgStd * rgStd + ybStd * ybStd) + 0.3 * sqrt(rgMean * rgMean + ybMean * ybMean)
    let warmCool = reds.reduce(0, +) / n - blues.reduce(0, +) / n

    var hist = [Int](repeating: 0, count: 32)
    for l in lum {
      let bin = min(31, max(0, Int(l * 32)))
      hist[bin] += 1
    }
    var entropy: Float = 0
    for c in hist where c > 0 {
      let p = Float(c) / n
      entropy -= p * log2(p)
    }

    let noiseMad = highPassMad(lum: lum, w: w, h: h)
    let q1 = Float(lum.filter { $0 < 0.25 }.count) / n
    let q2 = Float(lum.filter { $0 >= 0.25 && $0 < 0.5 }.count) / n
    let q3 = Float(lum.filter { $0 >= 0.5 && $0 < 0.75 }.count) / n
    let q4 = Float(lum.filter { $0 >= 0.75 }.count) / n

    return [
      meanL, stdL, Float(fracDark) / n, Float(fracBright) / n,
      lapVar, ratio, meanSat, stdSat, colorfulness, warmCool,
      entropy, noiseMad, q1, q2, q3, q4,
    ]
  }

  private static func laplacianVariance(lum: [Float], w: Int, h: Int) -> Float {
    guard w > 2, h > 2 else { return 0 }
    var vals = [Float]()
    for y in 1..<(h - 1) {
      for x in 1..<(w - 1) {
        let c = lum[y * w + x]
        let lap = -4 * c
          + lum[(y - 1) * w + x] + lum[(y + 1) * w + x]
          + lum[y * w + (x - 1)] + lum[y * w + (x + 1)]
        vals.append(lap)
      }
    }
    guard !vals.isEmpty else { return 0 }
    let m = vals.reduce(0, +) / Float(vals.count)
    var v: Float = 0
    for x in vals { v += (x - m) * (x - m) }
    return v / Float(vals.count)
  }

  private static func centerBorderLaplacian(lum: [Float], w: Int, h: Int) -> (Float, Float) {
    guard w > 2, h > 2 else { return (0, 0) }
    let y0 = h / 4, y1 = h - h / 4
    let x0 = w / 4, x1 = w - w / 4
    func lapAt(_ y: Int, _ x: Int) -> Float {
      let c = lum[y * w + x]
      return -4 * c
        + lum[(y - 1) * w + x] + lum[(y + 1) * w + x]
        + lum[y * w + (x - 1)] + lum[y * w + (x + 1)]
    }
    var center: [Float] = []
    var border: [Float] = []
    for y in 1..<(h - 1) {
      for x in 1..<(w - 1) {
        let v = lapAt(y, x)
        if y >= y0 && y < y1 && x >= x0 && x < x1 {
          center.append(v)
        } else {
          border.append(v)
        }
      }
    }
    func varOf(_ arr: [Float]) -> Float {
      guard !arr.isEmpty else { return 0 }
      let m = arr.reduce(0, +) / Float(arr.count)
      return arr.map { ($0 - m) * ($0 - m) }.reduce(0, +) / Float(arr.count)
    }
    return (varOf(center), varOf(border))
  }

  private static func highPassMad(lum: [Float], w: Int, h: Int) -> Float {
    guard w > 2, h > 2 else { return 0 }
    var high = [Float]()
    for y in 1..<(h - 1) {
      for x in 1..<(w - 1) {
        var sum: Float = 0
        for dy in -1...1 {
          for dx in -1...1 {
            sum += lum[(y + dy) * w + (x + dx)]
          }
        }
        let blur = sum / 9
        high.append(lum[y * w + x] - blur)
      }
    }
    guard !high.isEmpty else { return 0 }
    let med = high.sorted()[high.count / 2]
    let mad = high.map { abs($0 - med) }.sorted()[high.count / 2]
    return mad
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
