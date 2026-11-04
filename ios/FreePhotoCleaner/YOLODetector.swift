import Foundation
import CoreML
import Vision
import UIKit
import Photos

final class YOLODetector {
  static let shared = YOLODetector()

  private var vnModel: VNCoreMLModel?
  private let lock = NSLock()

  init() {
    lock.lock()
    defer { lock.unlock() }
    if let url = Bundle.main.url(forResource: "YOLOXNano", withExtension: "mlmodelc")
      ?? Bundle.main.url(forResource: "YOLOXNano", withExtension: "mlpackage") {
      do {
        let ml = try MLModel(contentsOf: url)
        vnModel = try VNCoreMLModel(for: ml)
      } catch {
        NSLog("[FeatureAI] YOLOX model load failed: %@", error.localizedDescription)
      }
    }
  }

  func detect(in image: UIImage) -> [DetectionBox] {
    lock.lock()
    let model = vnModel
    lock.unlock()
    guard let model = model, let cg = image.cgImage else { return [] }

    var boxes: [DetectionBox] = []
    let req = VNCoreMLRequest(model: model) { request, _ in
      guard let obs = request.results as? [VNCoreMLFeatureValueObservation],
            let multi = obs.first?.featureValue.multiArrayValue else { return }
      boxes = Self.parseFixedOutput(multi)
    }
    req.imageCropAndScaleOption = .scaleFill
    let handler = VNImageRequestHandler(cgImage: cg, orientation: .up, options: [:])
    try? handler.perform([req])
    return boxes
  }

  /// Output shape [1, 50, 6]: x1,y1,x2,y2,score,classId (classId -1 = pad)
  static func parseFixedOutput(_ array: MLMultiArray) -> [DetectionBox] {
    let shape = array.shape.map { $0.intValue }
    guard shape.count >= 2 else { return [] }
    let rows = shape[shape.count - 2]
    let cols = shape[shape.count - 1]
    guard cols >= 6 else { return [] }
    var out: [DetectionBox] = []
    for r in 0..<min(rows, FeatureSchema.maxBoxes) {
      let cid = read(array, r, 5)
      if cid < 0 { continue }
      let conf = read(array, r, 4)
      if conf < FeatureSchema.confThreshold { continue }
      out.append(DetectionBox(
        x1: read(array, r, 0),
        y1: read(array, r, 1),
        x2: read(array, r, 2),
        y2: read(array, r, 3),
        conf: conf,
        classId: Int(cid)
      ))
    }
    return out
  }

  private static func read(_ array: MLMultiArray, _ row: Int, _ col: Int) -> Float {
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
    guard let asset = fetch.firstObject else { return nil }

    let mod = asset.modificationDate?.timeIntervalSince1970 ?? 0
    let key = PhotoFeatures.cacheKey(localId: localId, modDate: mod)
    if let cached = PhotoFeatures.loadCachedFeatures(key: key) { return cached }

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
    PhotoFeatures.saveCachedFeatures(key: key, vector: vec)
    return vec
  }
}
