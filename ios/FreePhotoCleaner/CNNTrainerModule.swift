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
    if hasListeners { sendEvent(withName: name, body: body) }
  }

  private var modelsDir: URL {
    let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    let dir = docs.appendingPathComponent("ai_models", isDirectory: true)
    try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    return dir
  }

  private var activeIdURL: URL { modelsDir.appendingPathComponent("active_model_id.txt") }

  private func modelDir(_ modelId: String) -> URL {
    let dir = modelsDir.appendingPathComponent(modelId, isDirectory: true)
    try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    return dir
  }

  private func weightsURL(_ modelId: String) -> URL {
    modelDir(modelId).appendingPathComponent("weights.bin")
  }

  private func metaURL(_ modelId: String) -> URL {
    modelDir(modelId).appendingPathComponent("meta.json")
  }

  private func readActiveModelId() -> String? {
    guard let data = try? Data(contentsOf: activeIdURL),
          let s = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines),
          !s.isEmpty else { return nil }
    return s
  }

  private func writeActiveModelId(_ modelId: String) {
    try? modelId.data(using: .utf8)?.write(to: activeIdURL)
  }

  // MARK: - Bridge API

  @objc func isAvailable(_ resolve: RCTPromiseResolveBlock, rejecter reject: RCTPromiseRejectBlock) {
    resolve(true)
  }

  @objc func hasActiveModel(_ resolve: RCTPromiseResolveBlock, rejecter reject: RCTPromiseRejectBlock) {
    guard let id = readActiveModelId() else { resolve(false); return }
    resolve(FileManager.default.fileExists(atPath: weightsURL(id).path))
  }

  @objc func getActiveModelInfo(_ resolve: RCTPromiseResolveBlock, rejecter reject: RCTPromiseRejectBlock) {
    guard let id = readActiveModelId(),
          let data = try? Data(contentsOf: metaURL(id)),
          let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
      resolve(NSNull())
      return
    }
    resolve(json)
  }

  @objc func listModelCheckpoints(_ resolve: RCTPromiseResolveBlock, rejecter reject: RCTPromiseRejectBlock) {
    var list: [[String: Any]] = []
    guard let contents = try? FileManager.default.contentsOfDirectory(
      at: modelsDir, includingPropertiesForKeys: [.isDirectoryKey]
    ) else {
      resolve(list)
      return
    }
    for url in contents {
      var isDir: ObjCBool = false
      guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir), isDir.boolValue else { continue }
      let metaPath = url.appendingPathComponent("meta.json")
      guard let data = try? Data(contentsOf: metaPath),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { continue }
      list.append(json)
    }
    list.sort { (($0["trainedAt"] as? Double) ?? 0) > (($1["trainedAt"] as? Double) ?? 0) }
    resolve(list)
  }

  @objc func loadModel(_ modelId: String,
                       resolver resolve: RCTPromiseResolveBlock,
                       rejecter reject: RCTPromiseRejectBlock) {
    guard FileManager.default.fileExists(atPath: weightsURL(modelId).path) else {
      reject("NOT_FOUND", "No checkpoint for \(modelId)", nil)
      return
    }
    writeActiveModelId(modelId)
    activeModel = nil
    if ensureActiveModelLoaded() != nil {
      resolve(true)
    } else {
      reject("LOAD_FAIL", "Failed to load \(modelId)", nil)
    }
  }

  @objc func deleteModel(_ modelId: String,
                         resolver resolve: RCTPromiseResolveBlock,
                         rejecter reject: RCTPromiseRejectBlock) {
    try? FileManager.default.removeItem(at: modelDir(modelId))
    if readActiveModelId() == modelId {
      try? FileManager.default.removeItem(at: activeIdURL)
      activeModel = nil
      activeModelId = ""
    }
    resolve(true)
  }

  @objc func resetActiveModel(_ resolve: RCTPromiseResolveBlock, rejecter reject: RCTPromiseRejectBlock) {
    if let id = readActiveModelId() {
      try? FileManager.default.removeItem(at: modelDir(id))
    }
    try? FileManager.default.removeItem(at: activeIdURL)
    // legacy cleanup
    try? FileManager.default.removeItem(at: modelsDir.appendingPathComponent("active_weights.bin"))
    try? FileManager.default.removeItem(at: modelsDir.appendingPathComponent("active_meta.json"))
    activeModel = nil
    activeModelId = ""
    resolve(true)
  }

  @objc func stopTraining(_ resolve: RCTPromiseResolveBlock, rejecter reject: RCTPromiseRejectBlock) {
    stopRequested = true
    resolve(true)
  }

  @objc func startTraining(_ config: NSDictionary,
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
    let scheduler = (config["lrScheduler"] as? String) ?? "constant"
    let stepGamma = Float((config["stepGamma"] as? NSNumber)?.doubleValue ?? 0.5)
    let stepSize = max(1, (config["stepSize"] as? NSNumber)?.intValue ?? max(1, epochs / 3))
    let expDecay = Float((config["expDecay"] as? NSNumber)?.doubleValue ?? 0.95)
    let augs = parseAugmentations(config["augmentations"] as? NSDictionary)
    let labels = labelsRaw.map { Float($0.floatValue) }

    isTraining = true
    stopRequested = false

    trainQueue.async { [weak self] in
      guard let self = self else { return }
      do {
        let result = try self.runTraining(
          uris: uris, labels: labels, modelId: modelId, batchSize: batchSize,
          baseLr: lr, epochs: epochs, trainResize: trainResize, trainRatio: trainRatio,
          scheduler: scheduler, stepGamma: stepGamma, stepSize: stepSize, expDecay: expDecay,
          augs: augs
        )
        self.isTraining = false
        resolve(result)
      } catch {
        self.isTraining = false
        reject("TRAIN_ERROR", error.localizedDescription, error)
      }
    }
  }

  private struct AugConfig {
    var normalize = true
    var mean: [Float] = [0.485, 0.456, 0.406]
    var std: [Float] = [0.229, 0.224, 0.225]
    var randomCrop = false
    var cropProbability: Float = 0.5
    var cropFraction: Float = 0.85
    var randomFlip = true
    var flipProbability: Float = 0.5
    var randomRotation = false
    var rotationProbability: Float = 0.3
    var rotationDegrees: Float = 15
    var gaussianNoise = false
    var noiseStd: Float = 0.02
    var colorJitter = false
    var jitterDelta: Float = 0.1
    var randomGrayscale = false
    var grayscaleProbability: Float = 0.1
  }

  private func parseAugmentations(_ dict: NSDictionary?) -> AugConfig {
    var a = AugConfig()
    guard let d = dict else { return a }
    if let v = d["normalize"] as? Bool { a.normalize = v }
    if let m = d["normalizeMean"] as? [NSNumber], m.count == 3 {
      a.mean = m.map { Float($0.doubleValue) }
    }
    if let s = d["normalizeStd"] as? [NSNumber], s.count == 3 {
      a.std = s.map { Float($0.doubleValue) }
    }
    if let v = d["randomCrop"] as? Bool { a.randomCrop = v }
    if let v = d["cropProbability"] as? NSNumber { a.cropProbability = Float(v.doubleValue) }
    if let v = d["cropFraction"] as? NSNumber { a.cropFraction = Float(v.doubleValue) }
    if let v = d["randomFlip"] as? Bool { a.randomFlip = v }
    if let v = d["flipProbability"] as? NSNumber { a.flipProbability = Float(v.doubleValue) }
    if let v = d["randomRotation"] as? Bool { a.randomRotation = v }
    if let v = d["rotationProbability"] as? NSNumber { a.rotationProbability = Float(v.doubleValue) }
    if let v = d["rotationDegrees"] as? NSNumber { a.rotationDegrees = Float(v.doubleValue) }
    if let v = d["gaussianNoise"] as? Bool { a.gaussianNoise = v }
    if let v = d["noiseStd"] as? NSNumber { a.noiseStd = min(0.05, Float(v.doubleValue)) }
    if let v = d["colorJitter"] as? Bool { a.colorJitter = v }
    if let v = d["jitterDelta"] as? NSNumber { a.jitterDelta = Float(v.doubleValue) }
    if let v = d["randomGrayscale"] as? Bool { a.randomGrayscale = v }
    if let v = d["grayscaleProbability"] as? NSNumber { a.grayscaleProbability = Float(v.doubleValue) }
    return a
  }

  private struct TrainError: LocalizedError {
    let message: String
    var errorDescription: String? { message }
  }

  private func effectiveLr(base: Float, epoch: Int, epochs: Int, scheduler: String,
                           stepGamma: Float, stepSize: Int, expDecay: Float) -> Float {
    switch scheduler {
    case "cosine":
      let t = Float(epoch - 1) / Float(max(1, epochs - 1))
      return base * 0.5 * (1 + cos(Float.pi * t))
    case "step":
      let steps = (epoch - 1) / max(1, stepSize)
      return base * pow(stepGamma, Float(steps))
    case "exponential":
      return base * pow(expDecay, Float(epoch - 1))
    default:
      return base
    }
  }

  private func runTraining(
    uris: [String], labels: [Float], modelId: String, batchSize: Int,
    baseLr: Float, epochs: Int, trainResize: Int, trainRatio: Double,
    scheduler: String, stepGamma: Float, stepSize: Int, expDecay: Float,
    augs: AugConfig
  ) throws -> [String: Any] {
    if cacheResize != trainResize {
      tensorCache.removeAll()
      cacheResize = trainResize
    }

    // Decode to larger buffer for crop, then we'll resize in augment — store at trainResize for simplicity
    var samples: [(uri: String, tensor: Tensor, label: Float)] = []
    for (i, uri) in uris.enumerated() {
      if stopRequested { break }
      if let cached = tensorCache[uri] {
        samples.append((uri, cached, labels[i]))
        continue
      }
      if let tensor = decodeImage(uri: uri, size: trainResize, augs: augs, applyStochastic: false) {
        tensorCache[uri] = tensor
        samples.append((uri, tensor, labels[i]))
      }
    }

    guard samples.count >= 4 else {
      throw TrainError(message: "Not enough decodable images to train (need ≥ 4)")
    }

    var indices = Array(samples.indices)
    indices.shuffle()
    let trainCount = max(1, Int(Double(indices.count) * trainRatio))
    let trainIdx = Array(indices.prefix(trainCount))
    let testIdx = Array(indices.suffix(from: trainCount))
    let testIndices = testIdx.isEmpty ? Array(trainIdx.suffix(max(1, trainIdx.count / 5))) : testIdx

    var model = ModelFactory.create(modelId: modelId, resize: trainResize)
    var bestLoss = Float.greatestFiniteMagnitude
    var bestAcc: Float = 0
    var bestData: Data?
    var epochsRan = 0
    var cancelled = false

    for epoch in 1...epochs {
      if stopRequested { cancelled = true; break }
      epochsRan = epoch
      let lr = effectiveLr(base: baseLr, epoch: epoch, epochs: epochs, scheduler: scheduler,
                           stepGamma: stepGamma, stepSize: stepSize, expDecay: expDecay)
      model.zeroGrad()

      var trainLossSum: Float = 0
      var trainCorrect = 0
      var trainN = 0
      var shuffled = trainIdx
      shuffled.shuffle()
      var batchCount = 0

      for (batchPos, idx) in shuffled.enumerated() {
        if stopRequested { cancelled = true; break }
        let sample = samples[idx]
        let y = sample.label
        let x: Tensor
        if augs.randomCrop || augs.randomRotation {
          if let fresh = decodeImage(uri: sample.uri, size: trainResize, augs: augs, applyStochastic: true) {
            x = augmentTensor(fresh, augs: augs, stochastic: true)
          } else {
            x = augmentTensor(sample.tensor, augs: augs, stochastic: true)
          }
        } else {
          x = augmentTensor(sample.tensor, augs: augs, stochastic: true)
        }
        let pred = model.forward(x)
        trainLossSum += computeBCELoss(pred: pred, target: y)
        if (pred >= 0.5) == (y >= 0.5) { trainCorrect += 1 }
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
      var testCorrect = 0
      var testN = 0
      for idx in testIndices {
        let sample = samples[idx]
        let x = augmentTensor(sample.tensor, augs: augs, stochastic: false)
        let pred = model.forward(x)
        testLossSum += computeBCELoss(pred: pred, target: sample.label)
        if (pred >= 0.5) == (sample.label >= 0.5) { testCorrect += 1 }
        testN += 1
      }

      let trainLoss = trainLossSum / Float(max(1, trainN))
      let testLoss = testLossSum / Float(max(1, testN))
      let trainAcc = Float(trainCorrect) / Float(max(1, trainN))
      let testAcc = Float(testCorrect) / Float(max(1, testN))

      if testLoss < bestLoss {
        bestLoss = testLoss
        bestAcc = testAcc
        bestData = model.serialize()
        saveCheckpoint(model: model, modelId: modelId, resize: trainResize,
                       bestTestLoss: bestLoss, bestTestAcc: bestAcc,
                       testSize: testIndices.count, trainSize: trainIdx.count)
        activeModel = model
        activeModelId = modelId
        activeResize = trainResize
        writeActiveModelId(modelId)
      }

      emit("trainingProgress", body: [
        "epoch": epoch,
        "trainLoss": trainLoss,
        "testLoss": testLoss,
        "trainAcc": trainAcc,
        "testAcc": testAcc,
        "bestTestLoss": bestLoss,
        "bestTestAcc": bestAcc,
        "trainSize": trainIdx.count,
        "testSize": testIndices.count,
        "learningRate": lr,
        "done": false,
      ])
    }

    if let best = bestData {
      try? best.write(to: weightsURL(modelId))
      activeModel = nil
      activeModelId = modelId
      activeResize = trainResize
      writeActiveModelId(modelId)
      _ = ensureActiveModelLoaded()
    }

    let result: [String: Any] = [
      "epochsRan": epochsRan,
      "bestTestLoss": bestLoss == Float.greatestFiniteMagnitude ? NSNull() : bestLoss,
      "bestTestAcc": bestLoss == Float.greatestFiniteMagnitude ? NSNull() : bestAcc,
      "cancelled": cancelled,
      "trainSize": trainIdx.count,
      "testSize": testIndices.count,
      "modelId": modelId,
      "trainResize": trainResize,
    ]
    emit("trainingComplete", body: result)
    return result
  }

  private func saveCheckpoint(model: TrainableModel, modelId: String, resize: Int,
                              bestTestLoss: Float, bestTestAcc: Float,
                              testSize: Int, trainSize: Int) {
    let meta: [String: Any] = [
      "modelId": modelId,
      "trainResize": resize,
      "bestTestLoss": bestTestLoss,
      "bestTestAcc": bestTestAcc,
      "testSize": testSize,
      "trainSize": trainSize,
      "trainedAt": Date().timeIntervalSince1970 * 1000,
      "savedAt": Date().timeIntervalSince1970 * 1000,
    ]
    if let data = try? JSONSerialization.data(withJSONObject: meta) {
      try? data.write(to: metaURL(modelId))
    }
    try? model.serialize().write(to: weightsURL(modelId))
  }

  private func ensureActiveModelLoaded() -> TrainableModel? {
    if let m = activeModel { return m }
    guard let modelId = readActiveModelId() ?? {
      // migrate legacy active files
      let legacyMeta = modelsDir.appendingPathComponent("active_meta.json")
      let legacyWeights = modelsDir.appendingPathComponent("active_weights.bin")
      if let data = try? Data(contentsOf: legacyMeta),
         let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
         let id = json["modelId"] as? String {
        try? FileManager.default.copyItem(at: legacyWeights, to: weightsURL(id))
        try? data.write(to: metaURL(id))
        writeActiveModelId(id)
        return id
      }
      return nil
    }(),
    let metaData = try? Data(contentsOf: metaURL(modelId)),
    let meta = try? JSONSerialization.jsonObject(with: metaData) as? [String: Any],
    let resize = meta["trainResize"] as? Int,
    let weights = try? Data(contentsOf: weightsURL(modelId)) else {
      return nil
    }

    if modelId == "mobilenet-v3-head",
       let m = FrozenBackboneHeadModel.deserialize(weights, inputSize: resize) {
      activeModel = m; activeModelId = modelId; activeResize = resize
      return m
    }
    if let m = loadSequential(modelId: modelId, resize: resize, data: weights) {
      activeModel = m; activeModelId = modelId; activeResize = resize
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
              let w = readFloats(Int(o * i)), let b = readFloats(Int(o)),
              let linear = layer as? LinearLayer else { return nil }
        linear.weight = w; linear.bias = b
      case 2:
        guard let oc = readU32(), let ic = readU32(), let k = readU32(),
              let _ = readU32(), let _ = readU32(),
              let conv = layer as? Conv2dLayer else { return nil }
        guard let w = readFloats(Int(oc * ic * k * k)), let b = readFloats(Int(oc)) else { return nil }
        conv.weight = w; conv.bias = b
      case 3, 5, 6: continue
      case 4: _ = readU32(); _ = readU32()
      default: return nil
      }
    }
    return seq
  }

  @objc func predict(_ uris: [String],
                     resolver resolve: @escaping RCTPromiseResolveBlock,
                     rejecter reject: @escaping RCTPromiseRejectBlock) {
    trainQueue.async { [weak self] in
      guard let self = self else { return }
      guard let model = self.ensureActiveModelLoaded() else {
        reject("NO_MODEL", "No active trained model", nil)
        return
      }
      let resize = self.activeResize
      var augs = AugConfig()
      augs.normalize = true
      var scores: [Double] = []
      for uri in uris {
        if let tensor = self.tensorCache[uri] ?? self.decodeImage(uri: uri, size: resize, augs: augs, applyStochastic: false) {
          if self.tensorCache[uri] == nil { self.tensorCache[uri] = tensor }
          let x = self.augmentTensor(tensor, augs: augs, stochastic: false)
          scores.append(Double(model.forward(x)))
        } else {
          scores.append(0.5)
        }
      }
      resolve(scores)
    }
  }

  // MARK: - Augment / decode

  private func augmentTensor(_ input: Tensor, augs: AugConfig, stochastic: Bool) -> Tensor {
    var t = input.copy()
    let c = t.shape[0], h = t.shape[1], w = t.shape[2]
    let hw = h * w

    if stochastic && augs.randomFlip && Float.random(in: 0...1) < augs.flipProbability {
      for ch in 0..<c {
        for y in 0..<h {
          for x in 0..<(w / 2) {
            let i1 = (ch * h + y) * w + x
            let i2 = (ch * h + y) * w + (w - 1 - x)
            let tmp = t[i1]; t[i1] = t[i2]; t[i2] = tmp
          }
        }
      }
    }

    if stochastic && augs.randomGrayscale && Float.random(in: 0...1) < augs.grayscaleProbability {
      for i in 0..<hw {
        let gray = 0.299 * t[i] + 0.587 * t[hw + i] + 0.114 * t[2 * hw + i]
        t[i] = gray; t[hw + i] = gray; t[2 * hw + i] = gray
      }
    }

    if stochastic && augs.colorJitter {
      let d = augs.jitterDelta
      let br = Float.random(in: -d...d)
      let ct = 1 + Float.random(in: -d...d)
      for i in 0..<t.count {
        t[i] = (t[i] + br) * ct
      }
    }

    if stochastic && augs.gaussianNoise && augs.noiseStd > 0 {
      let std = min(0.05, augs.noiseStd)
      for i in 0..<t.count {
        // Box-Muller-ish approx
        let u1 = max(1e-6, Float.random(in: 0...1))
        let u2 = Float.random(in: 0...1)
        let n = sqrt(-2 * log(u1)) * cos(2 * Float.pi * u2) * std
        t[i] += n
      }
    }

    // Input tensors are stored as [-1,1] from decode; convert to [0,1] for normalize if needed
    if augs.normalize {
      for ch in 0..<min(3, c) {
        let mean = augs.mean[ch]
        let std = max(1e-6, augs.std[ch])
        let base = ch * hw
        for i in 0..<hw {
          // map [-1,1] -> [0,1] then normalize
          let p01 = (t[base + i] + 1) * 0.5
          t[base + i] = (p01 - mean) / std
        }
      }
    }

    // crop/rotation skipped on already-resized tensor for speed; crop applied at decode if needed
    _ = augs.randomCrop
    _ = augs.randomRotation
    return t
  }

  private func decodeImage(uri: String, size: Int, augs: AugConfig, applyStochastic: Bool) -> Tensor? {
    guard let localId = localIdentifier(from: uri) else { return nil }
    let fetchResult = PHAsset.fetchAssets(withLocalIdentifiers: [localId], options: nil)
    guard let asset = fetchResult.firstObject else { return nil }

    let options = PHImageRequestOptions()
    options.isSynchronous = true
    options.deliveryMode = .highQualityFormat
    options.resizeMode = .exact
    options.isNetworkAccessAllowed = true

    // Request larger if crop enabled
    let fetchSize = (augs.randomCrop && applyStochastic) ? Int(Float(size) / max(0.5, augs.cropFraction)) : size
    var resultImage: UIImage?
    PHImageManager.default().requestImage(
      for: asset,
      targetSize: CGSize(width: fetchSize, height: fetchSize),
      contentMode: .aspectFill,
      options: options
    ) { image, _ in resultImage = image }
    guard var image = resultImage else { return nil }

    if applyStochastic && augs.randomCrop && Float.random(in: 0...1) < augs.cropProbability {
      image = randomCrop(image, fraction: augs.cropFraction) ?? image
    }
    if applyStochastic && augs.randomRotation && Float.random(in: 0...1) < augs.rotationProbability {
      let angle = Float.random(in: -augs.rotationDegrees...augs.rotationDegrees) * .pi / 180
      image = rotate(image, radians: CGFloat(angle)) ?? image
    }

    return imageToTensor(image, size: size)
  }

  private func randomCrop(_ image: UIImage, fraction: Float) -> UIImage? {
    guard let cg = image.cgImage else { return nil }
    let w = cg.width, h = cg.height
    let fw = max(1, Int(Float(w) * fraction))
    let fh = max(1, Int(Float(h) * fraction))
    let x = Int.random(in: 0...max(0, w - fw))
    let y = Int.random(in: 0...max(0, h - fh))
    guard let cropped = cg.cropping(to: CGRect(x: x, y: y, width: fw, height: fh)) else { return nil }
    return UIImage(cgImage: cropped)
  }

  private func rotate(_ image: UIImage, radians: CGFloat) -> UIImage? {
    let size = image.size
    UIGraphicsBeginImageContextWithOptions(size, false, image.scale)
    guard let ctx = UIGraphicsGetCurrentContext() else { return nil }
    ctx.translateBy(x: size.width / 2, y: size.height / 2)
    ctx.rotate(by: radians)
    image.draw(in: CGRect(x: -size.width / 2, y: -size.height / 2, width: size.width, height: size.height))
    let out = UIGraphicsGetImageFromCurrentImageContext()
    UIGraphicsEndImageContext()
    return out
  }

  private func localIdentifier(from uri: String) -> String? {
    if uri.hasPrefix("ph://") {
      let rest = String(uri.dropFirst(5))
      if let slash = rest.firstIndex(of: "/") { return String(rest[..<slash]) }
      return rest
    }
    return uri
  }

  private func imageToTensor(_ image: UIImage, size: Int) -> Tensor? {
    guard let cgImage = image.cgImage else { return nil }
    let width = size, height = size
    let bytesPerRow = width * 4
    var rgba = [UInt8](repeating: 0, count: height * bytesPerRow)
    guard let ctx = CGContext(
      data: &rgba, width: width, height: height, bitsPerComponent: 8,
      bytesPerRow: bytesPerRow, space: CGColorSpaceCreateDeviceRGB(),
      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else { return nil }
    ctx.interpolationQuality = .medium
    ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
    let tensor = Tensor(shape: [3, height, width])
    let hw = height * width
    for y in 0..<height {
      for x in 0..<width {
        let pi = (y * width + x) * 4
        let idx = y * width + x
        tensor[idx] = Float(rgba[pi]) / 255.0 * 2 - 1
        tensor[hw + idx] = Float(rgba[pi + 1]) / 255.0 * 2 - 1
        tensor[2 * hw + idx] = Float(rgba[pi + 2]) / 255.0 * 2 - 1
      }
    }
    return tensor
  }
}
