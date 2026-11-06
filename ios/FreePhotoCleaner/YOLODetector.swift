import Foundation
import CoreML
import Vision
import UIKit
import Photos

// MARK: - YOLOX raw decode + NMS (Core ML exports [1, N, 85]: box offsets + sigmoid obj/cls; NMS runs on device)

private struct GridStride {
  let grid: [[Float]]  // [H*W][2] cell top-left (integer coords)
  let stride: Float
}

private struct RawCandidate {
  var x1: Float
  var y1: Float
  var x2: Float
  var y2: Float
  var score: Float
  var classId: Int
}

enum YOLOLoadStatus: String {
  case missing
  case loaded
  case failed
}

/// Core ML package + Swift postprocess. Raw output shape [1, numAnchors, 85] (YOLOX head, decode_in_inference=false).
final class YOLODetector {
  static let shared = YOLODetector()

  private var vnModel: VNCoreMLModel?
  private let lock = NSLock()

  private(set) var loadStatus: YOLOLoadStatus = .missing
  private(set) var loadErrorMessage: String?

  /// Last forward/decode issue (shape mismatch, etc.)
  private(set) var lastRunErrorMessage: String?

  private let gridStrides: [GridStride]
  private var totalAnchors: Int

  /// Feature schema v2: real YOLO (skip cache if model missing or decode fails).
  var shouldPersistFeatureCache: Bool {
    loadStatus == .loaded && lastRunErrorMessage == nil
  }

  init() {
    self.gridStrides = Self.buildGridStrides(
      inputSize: FeatureSchema.yoloLetterbox,
      strides: FeatureSchema.yoloStrides
    )
    self.totalAnchors = gridStrides.reduce(0) { $0 + $1.grid.count }

    lock.lock()
    defer { lock.unlock() }

    if let url = Bundle.main.url(forResource: "YOLOXNano", withExtension: "mlmodelc") {
      do {
        let ml = try MLModel(contentsOf: url)
        vnModel = try VNCoreMLModel(for: ml)
        loadStatus = .loaded
        loadErrorMessage = nil
      } catch {
        loadStatus = .failed
        loadErrorMessage = error.localizedDescription
        NSLog("[FeatureAI] YOLOX model load failed: %@", error.localizedDescription)
      }
    } else {
      loadStatus = .missing
      loadErrorMessage = "YOLOXNano.mlmodelc not found in app bundle (is YOLOXNano.mlpackage in the target's Compile Sources?)"
      NSLog("[FeatureAI] YOLOX model not bundled")
    }
  }

  func statusDictionary() -> [String: Any] {
    lock.lock()
    let st = loadStatus
    let le = loadErrorMessage
    let re = lastRunErrorMessage
    lock.unlock()
    return [
      "status": st.rawValue,
      "loadError": le ?? "",
      "lastRunError": re ?? "",
      "featureSchemaVersion": FeatureSchema.version,
      "expectedAnchors": totalAnchors,
    ]
  }

  func detect(in image: UIImage) -> [DetectionBox] {
    lock.lock()
    let model = vnModel
    lock.unlock()

    guard let model = model, let cg = image.cgImage else {
      lock.lock()
      lastRunErrorMessage = loadStatus == .loaded ? "No CGImage for detection" : nil
      lock.unlock()
      return []
    }

    lock.lock()
    lastRunErrorMessage = nil
    lock.unlock()

    var rawMulti: MLMultiArray?
    let req = VNCoreMLRequest(model: model) { request, error in
      if let error = error {
        NSLog("[FeatureAI] YOLOX Vision request error: %@", error.localizedDescription)
        return
      }
      if let obs = request.results as? [VNCoreMLFeatureValueObservation],
         let multi = obs.first?.featureValue.multiArrayValue {
        rawMulti = multi
        return
      }
      if let fv = (request.results as? [VNObservation])?.first as? VNCoreMLFeatureValueObservation,
         let multi = fv.featureValue.multiArrayValue {
        rawMulti = multi
      }
    }
    req.imageCropAndScaleOption = .scaleFill
    let handler = VNImageRequestHandler(cgImage: cg, orientation: .up, options: [:])
    try? handler.perform([req])

    guard let multi = rawMulti else {
      lock.lock()
      lastRunErrorMessage = "No Core ML output (VNCoreMLRequest)"
      lock.unlock()
      return []
    }

    let boxes = decodeRawOutput(multi, gridStrides: gridStrides, totalAnchors: totalAnchors)
    return boxes
  }

  // MARK: - Grid (matches YOLOXHead.decode_outputs)

  private static func buildGridStrides(inputSize: Int, strides: [Int]) -> [GridStride] {
    var out: [GridStride] = []
    for s in strides {
      let stride = Float(s)
      let h = inputSize / s
      let w = inputSize / s
      var rows: [[Float]] = []
      rows.reserveCapacity(h * w)
      for y in 0..<h {
        for x in 0..<w {
          rows.append([Float(x), Float(y)])
        }
      }
      out.append(GridStride(grid: rows, stride: stride))
    }
    return out
  }

  // MARK: - Decode [1, A, 85] → NMS → DetectionBox

  private func decodeRawOutput(_ array: MLMultiArray, gridStrides: [GridStride], totalAnchors: Int) -> [DetectionBox] {
    let shape = array.shape.map { $0.intValue }
    guard shape.count == 3,
          shape[0] == 1,
          shape[1] == totalAnchors,
          shape[2] == 85 else {
      let msg = "Unexpected YOLO raw shape \(shape); expected [1, \(totalAnchors), 85]"
      lock.lock()
      lastRunErrorMessage = msg
      lock.unlock()
      NSLog("[FeatureAI] %@", msg)
      return []
    }

    var candidates: [RawCandidate] = []
    candidates.reserveCapacity(256)

    var anchorIndex = 0
    for gs in gridStrides {
      for cell in gs.grid {
        guard anchorIndex < totalAnchors else { break }
        let gx = cell[0]
        let gy = cell[1]
        let tx = Self.readMulti(array, anchorIndex, 0)
        let ty = Self.readMulti(array, anchorIndex, 1)
        let tw = Self.readMulti(array, anchorIndex, 2)
        let th = Self.readMulti(array, anchorIndex, 3)
        // YOLOX's head applies sigmoid to obj/cls in eval mode, even with decode_in_inference=false.
        let obj = Self.readMulti(array, anchorIndex, 4)

        let cx = (tx + gx) * gs.stride
        let cy = (ty + gy) * gs.stride
        let w = exp(tw) * gs.stride
        let h = exp(th) * gs.stride

        if obj < FeatureSchema.confThreshold {
          anchorIndex += 1
          continue
        }

        var bestCls = 0
        var bestClsConf: Float = 0
        for c in 0..<80 {
          let p = Self.readMulti(array, anchorIndex, 5 + c)
          if p > bestClsConf {
            bestClsConf = p
            bestCls = c
          }
        }

        let score = obj * bestClsConf
        if score < FeatureSchema.confThreshold {
          anchorIndex += 1
          continue
        }

        let x1 = cx - w * 0.5
        let y1 = cy - h * 0.5
        let x2 = cx + w * 0.5
        let y2 = cy + h * 0.5

        let iw = Float(FeatureSchema.yoloLetterbox)
        let nx1 = max(0, min(1, x1 / iw))
        let ny1 = max(0, min(1, y1 / iw))
        let nx2 = max(0, min(1, x2 / iw))
        let ny2 = max(0, min(1, y2 / iw))

        candidates.append(RawCandidate(
          x1: nx1, y1: ny1, x2: nx2, y2: ny2,
          score: score, classId: bestCls
        ))

        anchorIndex += 1
      }
    }

    let kept = Self.nonMaxSuppressionClassAware(
      candidates: candidates,
      iouThreshold: FeatureSchema.yoloNmsThreshold,
      maxBoxes: FeatureSchema.maxBoxes
    )

    return kept.map {
      DetectionBox(
        x1: $0.x1, y1: $0.y1, x2: $0.x2, y2: $0.y2,
        conf: $0.score, classId: $0.classId
      )
    }
  }

  /// Greedy per-class NMS (same idea as torchvision `batched_nms`).
  private static func nonMaxSuppressionClassAware(
    candidates: [RawCandidate],
    iouThreshold: Float,
    maxBoxes: Int
  ) -> [RawCandidate] {
    let byClass = Dictionary(grouping: candidates, by: \.classId)
    var selected: [RawCandidate] = []
    selected.reserveCapacity(maxBoxes)

    for (_, group) in byClass {
      var remaining = group.sorted { $0.score > $1.score }
      while !remaining.isEmpty, selected.count < maxBoxes {
        let best = remaining.removeFirst()
        selected.append(best)
        remaining.removeAll { iou(a: best, b: $0) > iouThreshold }
      }
      if selected.count >= maxBoxes { break }
    }

    selected.sort { $0.score > $1.score }
    if selected.count > maxBoxes {
      selected = Array(selected.prefix(maxBoxes))
    }
    return selected
  }

  private static func iou(a: RawCandidate, b: RawCandidate) -> Float {
    let ix1 = max(a.x1, b.x1)
    let iy1 = max(a.y1, b.y1)
    let ix2 = min(a.x2, b.x2)
    let iy2 = min(a.y2, b.y2)
    let iw = max(0, ix2 - ix1)
    let ih = max(0, iy2 - iy1)
    let inter = iw * ih
    let areaA = max(0, a.x2 - a.x1) * max(0, a.y2 - a.y1)
    let areaB = max(0, b.x2 - b.x1) * max(0, b.y2 - b.y1)
    let union = areaA + areaB - inter + 1e-8
    return inter / union
  }

  private static func readMulti(_ array: MLMultiArray, _ row: Int, _ col: Int) -> Float {
    let idx: [NSNumber] = [0, NSNumber(value: row), NSNumber(value: col)]
    return array[idx].floatValue
  }
}

enum VisionFeatures {

  static func extract(from image: UIImage) -> (faces: [(CGRect, Float)], text: [CGRect]) {
    guard let cg = image.cgImage else { return ([], []) }
    var faces: [(CGRect, Float)] = []
    var texts: [CGRect] = []

    let faceReq = VNDetectFaceRectanglesRequest { req, _ in
      guard let results = req.results as? [VNFaceObservation] else { return }
      for f in results {
        faces.append((f.boundingBox, f.confidence))
      }
    }

    let textReq = VNDetectTextRectanglesRequest { req, _ in
      guard let results = req.results as? [VNTextObservation] else { return }
      for t in results {
        texts.append(t.boundingBox)
      }
    }
    textReq.reportCharacterBoxes = false

    let handler = VNImageRequestHandler(cgImage: cg, orientation: .up, options: [:])
    try? handler.perform([faceReq, textReq])
    return (faces, texts)
  }
}

final class FeatureExtractor {

  func extractVector(uri: String) -> [Float]? {
    guard let localId = PhotoFeatures.localIdentifier(from: uri) else { return nil }
    let fetch = PHAsset.fetchAssets(withLocalIdentifiers: [localId], options: nil)
    guard let asset = fetch.firstObject else {
      PhotoFeatures.deleteCachedFeatures(localId: localId)
      return nil
    }

    let mod = asset.modificationDate?.timeIntervalSince1970 ?? 0
    let key = PhotoFeatures.cacheKey(localId: localId, modDate: mod)
    if FeatureSchema.featureCacheEnabled, let cached = PhotoFeatures.loadCachedFeatures(key: key) {
      return cached
    }

    guard let decode = PhotoFeatures.decodeImage(asset: asset, longEdge: FeatureSchema.decodeLongEdge) else {
      return nil
    }
    let photoImg = PhotoFeatures.resizeLongest(decode, longest: FeatureSchema.photometricLongEdge) ?? decode
    let photo = PhotoFeatures.photometric(from: photoImg)

    let boxes = YOLODetector.shared.detect(in: decode)
    let yolo = PhotoFeatures.aggregateDetections(boxes)

    let (faceObs, textObs) = VisionFeatures.extract(from: decode)
    let vision = PhotoFeatures.visionFeatures(faces: faceObs, textRects: textObs)

    let fn = PhotoFeatures.filename(for: asset)
    let meta = PhotoFeatures.metadataVector(
      screenshot: PhotoFeatures.isScreenshot(asset: asset, filename: fn),
      whatsapp: PhotoFeatures.isWhatsAppFilename(fn),
      aspect: asset.pixelHeight > 0 ? Float(asset.pixelWidth) / Float(asset.pixelHeight) : 1
    )

    let vec = PhotoFeatures.assemble(photo: photo, yolo: yolo, vision: vision, meta: meta)
    guard vec.count == FeatureSchema.dim, vec.allSatisfy({ $0.isFinite }) else { return nil }
    if FeatureSchema.featureCacheEnabled, YOLODetector.shared.shouldPersistFeatureCache {
      PhotoFeatures.saveCachedFeatures(key: key, vector: vec)
    }
    return vec
  }
}
