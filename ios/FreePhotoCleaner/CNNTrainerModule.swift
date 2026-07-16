import Foundation
import Photos
import UIKit
import React

@objc(CNNTrainerModule)
class CNNTrainerModule: RCTEventEmitter {

  private var stopRequested = false
  private var isTraining = false
  private var tensorCache: [String: Tensor] = [:]
  private var cacheResize: Int = 0
  private var activeModel: TrainableModel?
  private var activeModelId: String = ""
  private var activeResize: Int = 64
  private let trainQueue = DispatchQueue(label: "com.freephotocleaner.cnntrainer", qos: .userInitiated)

  private var hasListeners = false

  override static func requiresMainQueueSetup() -> Bool { false }

  override func supportedEvents() -> [String]! {
    ["trainingProgress", "trainingComplete"]
  }

  override func startObserving() { hasListeners = true }
  override func stopObserving() { hasListeners = false }

  private func emit(_ name: String, body: [String: Any]) {
    if hasListeners {
      sendEvent(withName: name, body: body)
    }
  }

  private var modelsDir: URL {
    let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    let dir = docs.appendingPathComponent("ai_models", isDirectory: true)
    try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    return dir
  }

  private var metaURL: URL { modelsDir.appendingPathComponent("active_meta.json") }
  private var weightsURL: URL { modelsDir.appendingPathComponent("active_weights.bin") }

  @objc
  func isAvailable(_ resolve: RCTPromiseResolveBlock, rejecter reject: RCTPromiseRejectBlock) {
    resolve(true)
  }

  @objc
  func hasActiveModel(_ resolve: RCTPromiseResolveBlock, rejecter reject: RCTPromiseRejectBlock) {
    resolve(FileManager.default.fileExists(atPath: weightsURL.path) &&
            FileManager.default.fileExists(atPath: metaURL.path))
  }

  @objc
  func getActiveModelInfo(_ resolve: RCTPromiseResolveBlock, rejecter reject: RCTPromiseRejectBlock) {
    guard let data = try? Data(contentsOf: metaURL),
          let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
      resolve(NSNull())
      return
    }
    resolve(json)
  }

  @objc
  func resetActiveModel(_ resolve: RCTPromiseResolveBlock, rejecter reject: RCTPromiseRejectBlock) {
    try? FileManager.default.removeItem(at: weightsURL)
    try? FileManager.default.removeItem(at: metaURL)
    activeModel = nil
    resolve(true)
  }

  @objc
  func stopTraining(_ resolve: RCTPromiseResolveBlock, rejecter reject: RCTPromiseRejectBlock) {
    stopRequested = true
    resolve(true)
  }

  @objc
  func startTraining(_ config: NSDictionary,
                     resolver resolve: @escaping RCTPromiseResolveBlock,
                     rejecter reject: @escaping RCTPromiseRejectBlock) {
    if isTraining {
      reject("BUSY", "Training already in progress", nil)
      return
    }

    guard let uris = config["uris"] as? [String],
          let labelsRaw = config["labels"] as? [NSNumber],
          uris.count == labelsRaw.count,
          uris.count > 0 else {
      reject("INVALID", "uris/labels required and must match", nil)
      return
    }

    let modelId = (config["modelId"] as? String) ?? "cnn-nano"
    let batchSize = max(1, (config["batchSize"] as? NSNumber)?.intValue ?? 8)
    let lr = Float((config["learningRate"] as? NSNumber)?.doubleValue ?? 0.01)
    let epochs = max(1, (config["epochs"] as? NSNumber)?.intValue ?? 10)
    let trainResize = clampResize((config["trainResize"] as? NSNumber)?.intValue ?? 64)
    let trainRatio = (config["trainRatio"] as? NSNumber)?.doubleValue ?? 0.8

    let labels = labelsRaw.map { Float($0.floatValue) }

    isTraining = true
    stopRequested = false

    trainQueue.async { [weak self] in
      guard let self = self else { return }
      do {
        let result = try self.runTraining(
          uris: uris,
          labels: labels,
          modelId: modelId,
          batchSize: batchSize,
          lr: lr,
          epochs: epochs,
          trainResize: trainResize,
          trainRatio: trainRatio
        )
        self.isTraining = false
        resolve(result)
      } catch {
        self.isTraining = false
        reject("TRAIN_ERROR", error.localizedDescription, error)
      }
    }
  }

  private struct TrainError: LocalizedError {
    let message: String
    var errorDescription: String? { message }
  }

  private func runTraining(
    uris: [String],
    labels: [Float],
    modelId: String,
    batchSize: Int,
    lr: Float,
    epochs: Int,
    trainResize: Int,
    trainRatio: Double
  ) throws -> [String: Any] {
    // Decode images
    if cacheResize != trainResize {
      tensorCache.removeAll()
      cacheResize = trainResize
    }

    var samples: [(Tensor, Float)] = []
    for (i, uri) in uris.enumerated() {
      if stopRequested { break }
      if let cached = tensorCache[uri] {
        samples.append((cached, labels[i]))
        continue
      }
      if let tensor = decodeImage(uri: uri, size: trainResize) {
        tensorCache[uri] = tensor
        samples.append((tensor, labels[i]))
      }
    }

    guard samples.count >= 4 else {
      throw TrainError(message: "Not enough decodable images to train (need ≥ 4)")
    }

    // Shuffle + split
    var indices = Array(samples.indices)
    indices.shuffle()
    let trainCount = max(1, Int(Double(indices.count) * trainRatio))
    let trainIdx = Array(indices.prefix(trainCount))
    let testIdx = Array(indices.suffix(from: trainCount))
    let testIndices = testIdx.isEmpty ? Array(trainIdx.suffix(max(1, trainIdx.count / 5))) : testIdx

    var model = ModelFactory.create(modelId: modelId, resize: trainResize)
    var bestLoss = Float.greatestFiniteMagnitude
    var bestData: Data?
    var epochsRan = 0
    var cancelled = false

    for epoch in 1...epochs {
      if stopRequested {
        cancelled = true
        break
      }
      epochsRan = epoch
      model.zeroGrad()

      var trainLossSum: Float = 0
      var trainN = 0
      var shuffled = trainIdx
      shuffled.shuffle()

      var batchCount = 0
      for (batchPos, idx) in shuffled.enumerated() {
        if stopRequested {
          cancelled = true
          break
        }
        let (x, y) = samples[idx]
        let pred = model.forward(x)
        trainLossSum += computeBCELoss(pred: pred, target: y)
        trainN += 1
        model.backward(target: y)
        batchCount += 1
        if batchCount >= batchSize || batchPos == shuffled.count - 1 {
          model.applyGrad(lr: lr, batchSize: batchCount)
          model.zeroGrad()
          batchCount = 0
        }
      }

      if cancelled { break }

      var testLossSum: Float = 0
      var testN = 0
      for idx in testIndices {
        let (x, y) = samples[idx]
        let pred = model.forward(x)
        testLossSum += computeBCELoss(pred: pred, target: y)
        testN += 1
      }

      let trainLoss = trainLossSum / Float(max(1, trainN))
      let testLoss = testLossSum / Float(max(1, testN))

      if testLoss < bestLoss {
        bestLoss = testLoss
        bestData = model.serialize()
        // Keep best in memory by also cloning via re-create + we'll save bestData
        saveActive(model: model, modelId: modelId, resize: trainResize, bestTestLoss: bestLoss)
        activeModel = model
        activeModelId = modelId
        activeResize = trainResize
      }

      emit("trainingProgress", body: [
        "epoch": epoch,
        "trainLoss": trainLoss,
        "testLoss": testLoss,
        "bestTestLoss": bestLoss,
        "trainSize": trainIdx.count,
        "testSize": testIndices.count,
        "done": false,
      ])
    }

    if let best = bestData {
      try? best.write(to: weightsURL)
      // Reload best checkpoint into memory (training may have continued past best)
      activeModel = nil
      activeModelId = modelId
      activeResize = trainResize
      _ = ensureActiveModelLoaded()
    }

    let result: [String: Any] = [
      "epochsRan": epochsRan,
      "bestTestLoss": bestLoss == Float.greatestFiniteMagnitude ? NSNull() : bestLoss,
      "cancelled": cancelled,
      "trainSize": trainIdx.count,
      "testSize": testIndices.count,
      "modelId": modelId,
      "trainResize": trainResize,
    ]

    emit("trainingComplete", body: result)
    return result
  }

  private func saveActive(model: TrainableModel, modelId: String, resize: Int, bestTestLoss: Float) {
    let meta: [String: Any] = [
      "modelId": modelId,
      "trainResize": resize,
      "bestTestLoss": bestTestLoss,
      "savedAt": Date().timeIntervalSince1970 * 1000,
    ]
    if let data = try? JSONSerialization.data(withJSONObject: meta) {
      try? data.write(to: metaURL)
    }
    try? model.serialize().write(to: weightsURL)
  }

  private func ensureActiveModelLoaded() -> TrainableModel? {
    if let m = activeModel { return m }
    guard let metaData = try? Data(contentsOf: metaURL),
          let meta = try? JSONSerialization.jsonObject(with: metaData) as? [String: Any],
          let modelId = meta["modelId"] as? String,
          let resize = meta["trainResize"] as? Int,
          let weights = try? Data(contentsOf: weightsURL) else {
      return nil
    }

    if modelId == "resnet-18" {
      if let m = ResNet18Model.deserialize(weights, inputSize: resize) {
        activeModel = m
        activeModelId = modelId
        activeResize = resize
        return m
      }
    }

    // Rebuild sequential and load tagged weights
    if let m = loadSequential(modelId: modelId, resize: resize, data: weights) {
      activeModel = m
      activeModelId = modelId
      activeResize = resize
      return m
    }
    return nil
  }

  private func loadSequential(modelId: String, resize: Int, data: Data) -> SequentialModel? {
    let model = ModelFactory.create(modelId: modelId, resize: resize)
    guard let seq = model as? SequentialModel else { return nil }
    var offset = 0

    func readU32() -> UInt32? {
      guard offset + 4 <= data.count else { return nil }
      let v = data.subdata(in: offset..<(offset + 4)).withUnsafeBytes { $0.load(as: UInt32.self) }
      offset += 4
      return v
    }
    func readFloats(_ n: Int) -> [Float]? {
      let byteCount = n * 4
      guard offset + byteCount <= data.count else { return nil }
      let slice = data.subdata(in: offset..<(offset + byteCount))
      offset += byteCount
      return slice.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    guard let idLen = readU32(), offset + Int(idLen) <= data.count else { return nil }
    offset += Int(idLen)

    for layer in seq.layers {
      guard let tag = readU32() else { return nil }
      if tag == 0 { break }
      switch tag {
      case 1:
        guard let o = readU32(), let i = readU32(),
              let w = readFloats(Int(o * i)),
              let b = readFloats(Int(o)),
              let linear = layer as? LinearLayer else { return nil }
        linear.weight = w
        linear.bias = b
      case 2:
        guard let oc = readU32(), let ic = readU32(), let k = readU32(),
              let _ = readU32(), let _ = readU32(),
              let conv = layer as? Conv2dLayer else { return nil }
        let wCount = Int(oc * ic * k * k)
        guard let w = readFloats(wCount), let b = readFloats(Int(oc)) else { return nil }
        conv.weight = w
        conv.bias = b
      case 3, 5, 6:
        continue
      case 4:
        _ = readU32(); _ = readU32()
      default:
        return nil
      }
    }
    return seq
  }

  @objc
  func predict(_ uris: [String],
               resolver resolve: @escaping RCTPromiseResolveBlock,
               rejecter reject: @escaping RCTPromiseRejectBlock) {
    trainQueue.async { [weak self] in
      guard let self = self else { return }
      guard let model = self.ensureActiveModelLoaded() else {
        reject("NO_MODEL", "No active trained model", nil)
        return
      }
      let resize = self.activeResize
      var scores: [Double] = []
      for uri in uris {
        if let tensor = self.tensorCache[uri] ?? self.decodeImage(uri: uri, size: resize) {
          if self.tensorCache[uri] == nil { self.tensorCache[uri] = tensor }
          scores.append(Double(model.forward(tensor)))
        } else {
          scores.append(0.5)
        }
      }
      resolve(scores)
    }
  }

  // MARK: - Image decode

  private func decodeImage(uri: String, size: Int) -> Tensor? {
    let localId = localIdentifier(from: uri)
    let fetchResult: PHFetchResult<PHAsset>
    if let localId = localId {
      fetchResult = PHAsset.fetchAssets(withLocalIdentifiers: [localId], options: nil)
    } else {
      return nil
    }
    guard let asset = fetchResult.firstObject else { return nil }

    let options = PHImageRequestOptions()
    options.isSynchronous = true
    options.deliveryMode = .highQualityFormat
    options.resizeMode = .exact
    options.isNetworkAccessAllowed = true

    var resultImage: UIImage?
    let target = CGSize(width: size, height: size)
    PHImageManager.default().requestImage(
      for: asset,
      targetSize: target,
      contentMode: .aspectFill,
      options: options
    ) { image, _ in
      resultImage = image
    }
    guard let image = resultImage else { return nil }
    return imageToTensor(image, size: size)
  }

  private func localIdentifier(from uri: String) -> String? {
    // ph://UUID/L0/001 or similar
    if uri.hasPrefix("ph://") {
      let rest = String(uri.dropFirst(5))
      if let slash = rest.firstIndex(of: "/") {
        return String(rest[..<slash])
      }
      return rest
    }
    if uri.contains("/") {
      // already a local identifier style
      return uri
    }
    return uri
  }

  private func imageToTensor(_ image: UIImage, size: Int) -> Tensor? {
    guard let cgImage = image.cgImage else { return nil }
    let width = size
    let height = size
    let bytesPerRow = width * 4
    var rgba = [UInt8](repeating: 0, count: height * bytesPerRow)
    guard let ctx = CGContext(
      data: &rgba,
      width: width,
      height: height,
      bitsPerComponent: 8,
      bytesPerRow: bytesPerRow,
      space: CGColorSpaceCreateDeviceRGB(),
      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else { return nil }

    ctx.interpolationQuality = .medium
    ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

    let tensor = Tensor(shape: [3, height, width])
    let hw = height * width
    for y in 0..<height {
      for x in 0..<width {
        let pi = (y * width + x) * 4
        let r = Float(rgba[pi]) / 255.0
        let g = Float(rgba[pi + 1]) / 255.0
        let b = Float(rgba[pi + 2]) / 255.0
        // ImageNet-ish normalize optional; use simple [-1,1]
        let idx = y * width + x
        tensor[idx] = r * 2 - 1
        tensor[hw + idx] = g * 2 - 1
        tensor[2 * hw + idx] = b * 2 - 1
      }
    }
    return tensor
  }
}
