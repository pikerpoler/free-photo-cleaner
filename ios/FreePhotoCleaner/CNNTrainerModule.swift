import Foundation
import Photos
import UIKit
import React

@objc(CNNTrainerModule)
class CNNTrainerModule: RCTEventEmitter {

  private var stopRequested = false
  private var isTraining = false
  private let trainQueue = DispatchQueue(label: "com.freephotocleaner.featuretrainer", qos: .userInitiated)
  private var hasListeners = false
  private let extractor = FeatureExtractor()
  private let bank = FeatureModelBank()

  override static func requiresMainQueueSetup() -> Bool { false }

  override func supportedEvents() -> [String]! {
    ["trainingProgress", "trainingComplete"]
  }

  override func startObserving() { hasListeners = true }
  override func stopObserving() { hasListeners = false }

  private func emit(_ name: String, body: [String: Any]) {
    let send = { [weak self] in
      guard let self = self, self.hasListeners else { return }
      self.sendEvent(withName: name, body: body)
    }
    if Thread.isMainThread { send() } else { DispatchQueue.main.async(execute: send) }
  }

  private func finish(_ resolve: @escaping RCTPromiseResolveBlock, _ value: Any?) {
    if Thread.isMainThread { resolve(value) } else { DispatchQueue.main.async { resolve(value) } }
  }

  private func fail(_ reject: @escaping RCTPromiseRejectBlock, _ code: String, _ message: String) {
    if Thread.isMainThread {
      reject(code, message, nil)
    } else {
      DispatchQueue.main.async { reject(code, message, nil) }
    }
  }

  // MARK: - Bridge API

  @objc func isAvailable(_ resolve: RCTPromiseResolveBlock, rejecter reject: RCTPromiseRejectBlock) {
    resolve(true)
  }

  @objc func hasActiveModel(_ resolve: RCTPromiseResolveBlock, rejecter reject: RCTPromiseRejectBlock) {
    resolve(bank.hasAnyModel())
  }

  @objc func getActiveModelInfo(_ resolve: RCTPromiseResolveBlock, rejecter reject: RCTPromiseRejectBlock) {
    resolve(bank.checkpointMeta(for: bank.activeModelId))
  }

  @objc func listModelCheckpoints(_ resolve: RCTPromiseResolveBlock, rejecter reject: RCTPromiseRejectBlock) {
    resolve(bank.listCheckpoints())
  }

  @objc func loadModel(_ modelId: String,
                       resolver resolve: RCTPromiseResolveBlock,
                       rejecter reject: RCTPromiseRejectBlock) {
    guard FeatureSchema.scoreModelIds.contains(modelId) else {
      reject("NOT_FOUND", "Unknown model \(modelId)", nil)
      return
    }
    bank.setActive(modelId)
    resolve(true)
  }

  @objc func deleteModel(_ modelId: String,
                         resolver resolve: RCTPromiseResolveBlock,
                         rejecter reject: RCTPromiseRejectBlock) {
    guard FeatureSchema.scoreModelIds.contains(modelId) else {
      reject("NOT_FOUND", "Unknown model \(modelId)", nil)
      return
    }
    bank.deleteModel(modelId)
    resolve(true)
  }

  @objc func getModelStatus(_ modelId: String,
                            resolver resolve: RCTPromiseResolveBlock,
                            rejecter reject: RCTPromiseRejectBlock) {
    let id = FeatureSchema.scoreModelIds.contains(modelId) ? modelId : bank.activeModelId
    resolve(bank.status(for: id))
  }

  @objc func resetActiveModel(_ resolve: RCTPromiseResolveBlock, rejecter reject: RCTPromiseRejectBlock) {
    bank.resetAll()
    resolve(true)
  }

  @objc func stopTraining(_ resolve: RCTPromiseResolveBlock, rejecter reject: RCTPromiseRejectBlock) {
    stopRequested = true
    resolve(true)
  }

  @objc func extractFeatures(_ uri: String,
                             resolver resolve: @escaping RCTPromiseResolveBlock,
                             rejecter reject: @escaping RCTPromiseRejectBlock) {
    trainQueue.async { [weak self] in
      guard let self = self else { return }
      guard YOLODetector.shared.shouldPersistFeatureCache else {
        self.finish(resolve, NSNull())
        return
      }
      if let vec = self.extractor.extractVector(uri: uri), vec.count == FeatureSchema.dim {
        self.finish(resolve, vec.map { Double($0) })
      } else {
        self.finish(resolve, NSNull())
      }
    }
  }

  @objc func startTraining(_ config: NSDictionary,
                           resolver resolve: @escaping RCTPromiseResolveBlock,
                           rejecter reject: @escaping RCTPromiseRejectBlock) {
    guard !isTraining else {
      reject("BUSY", "Training already in progress", nil)
      return
    }
    guard let featureRows = config["features"] as? [Any],
          let labels = config["labels"] as? [NSNumber] else {
      reject("BAD_CONFIG", "Missing features/labels", nil)
      return
    }
    let requested = (config["modelId"] as? String) ?? bank.activeModelId
    let modelId = FeatureSchema.scoreModelIds.contains(requested) ? requested : bank.activeModelId
    bank.setActive(modelId)
    let mlpHyper = MLPHyperparams(
      batchSize: (config["batchSize"] as? NSNumber)?.intValue ?? 8,
      learningRate: (config["learningRate"] as? NSNumber)?.floatValue ?? 0.001,
      scheduler: (config["scheduler"] as? String) ?? "constant",
      stepGamma: (config["stepGamma"] as? NSNumber)?.floatValue ?? 0.5,
      stepEvery: (config["stepEvery"] as? NSNumber)?.intValue ?? 5,
      expDecay: (config["expDecay"] as? NSNumber)?.floatValue ?? 0.95,
      maxEpochs: (config["maxEpochs"] as? NSNumber)?.intValue ?? 80
    )
    let logisticHyper = LogisticHyperparams(
      C: (config["logisticC"] as? NSNumber)?.floatValue ?? 1,
      maxIter: (config["logisticMaxIter"] as? NSNumber)?.intValue ?? 500
    )
    let catboostHyper = CatBoostHyperparams(
      learningRate: (config["catboostLearningRate"] as? NSNumber)?.floatValue ?? 0.1,
      maxDepth: (config["catboostDepth"] as? NSNumber)?.intValue ?? 3,
      minLeaf: (config["catboostMinLeaf"] as? NSNumber)?.intValue ?? 8,
      maxNewTrees: (config["catboostMaxNewTrees"] as? NSNumber)?.intValue ?? 40
    )

    isTraining = true
    stopRequested = false

    trainQueue.async { [weak self] in
      guard let self = self else { return }
      defer {
        self.bank.discardPending()
        self.isTraining = false
      }

      var rows: [[Float]] = []
      var y: [Float] = []
      let count = min(featureRows.count, labels.count)
      var unreadable = 0
      let tooFewMessage =
        "Training needs at least 20 saved photos, including at least 2 kept and 2 marked for deletion."

      for idx in 0..<count {
        if self.stopRequested { break }
        guard let nums = featureRows[idx] as? [NSNumber], nums.count == FeatureSchema.dim else {
          unreadable += 1
          continue
        }
        rows.append(nums.map { $0.floatValue })
        y.append(Float(truncating: labels[idx]))
      }

      if self.stopRequested {
        let result: [String: Any] = [
          "epochsRan": 0,
          "bestTestLoss": NSNull(),
          "bestTestAcc": NSNull(),
          "accuracy": NSNull(),
          "f1": NSNull(),
          "cancelled": true,
          "trainSize": 0,
          "testSize": 0,
          "skippedUnreadable": unreadable,
          "modelId": modelId,
          "trainResize": 0,
        ]
        self.emit("trainingComplete", body: result)
        self.finish(resolve, result)
        return
      }

      if rows.count < 20 {
        self.fail(reject, "TOO_FEW", tooFewMessage)
        return
      }
      let nKeep = y.filter { $0 < 0.5 }.count
      let nDel = y.count - nKeep
      if nKeep < 2 || nDel < 2 {
        self.fail(reject, "TOO_FEW", tooFewMessage)
        return
      }

      let seed = Int(Date().timeIntervalSince1970) % 10_000
      guard let split = stratifiedInnerSplit(labels: y, seed: seed) else {
        self.fail(reject, "TOO_FEW", tooFewMessage)
        return
      }
      let trainIdx = split.0
      let testIdx = split.1

      let metrics = self.bank.trainReplacement(
        modelId: modelId,
        rows: rows,
        labels: y,
        trainIdx: trainIdx,
        testIdx: testIdx,
        seed: seed,
        mlpHyper: mlpHyper,
        logisticHyper: logisticHyper,
        catboostHyper: catboostHyper,
        shouldStop: { self.stopRequested },
        onEpoch: { epoch, trainLoss, valLoss, trainAcc, valAcc, lr in
          self.emit("trainingProgress", body: [
            "phase": "fit",
            "epoch": epoch,
            "trainLoss": Double(trainLoss),
            "testLoss": Double(valLoss),
            "trainAcc": Double(trainAcc),
            "testAcc": Double(valAcc),
            "bestTestLoss": Double(valLoss),
            "bestTestAcc": Double(valAcc),
            "trainSize": trainIdx.count,
            "testSize": testIdx.count,
            "learningRate": Double(lr),
            "done": false,
          ])
        }
      )

      if self.stopRequested {
        self.bank.discardPending()
        let result: [String: Any] = [
          "epochsRan": metrics.epoch,
          "bestTestLoss": NSNull(),
          "bestTestAcc": NSNull(),
          "accuracy": NSNull(),
          "f1": NSNull(),
          "cancelled": true,
          "trainSize": trainIdx.count,
          "testSize": testIdx.count,
          "skippedUnreadable": unreadable,
          "modelId": modelId,
          "trainResize": 0,
        ]
        self.emit("trainingComplete", body: result)
        self.finish(resolve, result)
        return
      }

      do {
        try self.bank.saveSelected(
          modelId: modelId,
          metrics: metrics,
          trainSize: trainIdx.count,
          testSize: testIdx.count
        )
      } catch {
        self.bank.discardPending()
        self.fail(reject, "SAVE_FAILED", error.localizedDescription)
        return
      }

      let result: [String: Any] = [
        "epochsRan": metrics.epoch,
        "bestTestLoss": metrics.testLoss,
        "bestTestAcc": metrics.testAcc,
        "accuracy": metrics.testAcc,
        "f1": metrics.f1,
        "cancelled": false,
        "trainSize": trainIdx.count,
        "testSize": testIdx.count,
        "skippedUnreadable": unreadable,
        "modelId": modelId,
        "trainResize": 0,
      ]
      self.emit("trainingComplete", body: result)
      self.finish(resolve, result)
    }
  }

  @objc func predict(_ uris: [String],
                     resolver resolve: @escaping RCTPromiseResolveBlock,
                     rejecter reject: @escaping RCTPromiseRejectBlock) {
    trainQueue.async { [weak self] in
      guard let self = self else { return }
      guard self.bank.hasAnyModel() else {
        self.fail(reject, "NO_MODEL", "No trained model")
        return
      }
      var scores: [Double] = []
      for uri in uris {
        if let vec = self.extractor.extractVector(uri: uri) {
          let p = self.bank.predict(vec, modelId: self.bank.activeModelId)
          scores.append(Double(p))
        } else {
          scores.append(0.5)
        }
      }
      self.finish(resolve, scores)
    }
  }

  @objc func getYOLOStatus(_ resolve: RCTPromiseResolveBlock, rejecter reject: RCTPromiseRejectBlock) {
    resolve(YOLODetector.shared.statusDictionary())
  }

  /// Remove on-disk feature vectors for deleted photos (ph:// URIs).
  @objc func deleteCachedFeatures(_ uris: [String],
                                  resolver resolve: @escaping RCTPromiseResolveBlock,
                                  rejecter reject: @escaping RCTPromiseRejectBlock) {
    trainQueue.async { [weak self] in
      guard let self = self else { return }
      PhotoFeatures.deleteCachedFeatures(photoUris: uris)
      self.finish(resolve, true)
    }
  }

  /// Keep only caches for photos still in the library (ph:// URIs from catalog sync).
  @objc func pruneFeatureCache(_ uris: [String],
                               resolver resolve: @escaping RCTPromiseResolveBlock,
                               rejecter reject: @escaping RCTPromiseRejectBlock) {
    trainQueue.async { [weak self] in
      guard let self = self else { return }
      PhotoFeatures.pruneFeatureCache(photoUris: uris)
      self.finish(resolve, true)
    }
  }
}
