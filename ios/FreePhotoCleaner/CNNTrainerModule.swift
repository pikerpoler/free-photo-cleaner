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

  @objc func startTraining(_ config: NSDictionary,
                           resolver resolve: @escaping RCTPromiseResolveBlock,
                           rejecter reject: @escaping RCTPromiseRejectBlock) {
    guard !isTraining else {
      reject("BUSY", "Training already in progress", nil)
      return
    }
    guard let uris = config["uris"] as? [String],
          let labels = config["labels"] as? [NSNumber] else {
      reject("BAD_CONFIG", "Missing uris/labels", nil)
      return
    }
    let requested = (config["modelId"] as? String) ?? bank.activeModelId
    let modelId = FeatureSchema.scoreModelIds.contains(requested) ? requested : bank.activeModelId
    bank.setActive(modelId)
    let hyper = MLPHyperparams(
      batchSize: (config["batchSize"] as? NSNumber)?.intValue ?? 8,
      learningRate: (config["learningRate"] as? NSNumber)?.floatValue ?? 0.001,
      scheduler: (config["scheduler"] as? String) ?? "constant",
      stepGamma: (config["stepGamma"] as? NSNumber)?.floatValue ?? 0.5,
      stepEvery: (config["stepEvery"] as? NSNumber)?.intValue ?? 5,
      expDecay: (config["expDecay"] as? NSNumber)?.floatValue ?? 0.95,
      maxEpochs: 40
    )

    isTraining = true
    stopRequested = false

    trainQueue.async { [weak self] in
      guard let self = self else { return }
      defer { self.isTraining = false }

      var sampleIds: [String] = []
      var rows: [[Float]] = []
      var y: [Float] = []
      let count = min(uris.count, labels.count)
      let alreadyTrained = self.bank.hasModel(modelId)

      for idx in 0..<count {
        if self.stopRequested { break }
        let uri = uris[idx]
        let label = Float(truncating: labels[idx])
        guard let localId = PhotoFeatures.localIdentifier(from: uri) else { continue }
        if self.bank.isIncorporated(localId, modelId: modelId) { continue }

        self.emit("trainingProgress", body: [
          "phase": "features",
          "index": idx + 1,
          "total": count,
          "done": false,
        ])

        guard let vec = self.extractor.extractVector(uri: uri) else { continue }
        sampleIds.append(localId)
        rows.append(vec)
        y.append(label)
      }

      if rows.isEmpty {
        self.fail(
          reject,
          "NO_NEW",
          "This model already includes every labeled photo. Training only adds photos it has not seen. Delete the model at the bottom of this screen to train it from scratch."
        )
        return
      }
      if !alreadyTrained && rows.count < 20 {
        self.fail(reject, "TOO_FEW", "Need at least 20 labeled photos to start this model")
        return
      }
      let nKeep = y.filter { $0 < 0.5 }.count
      let nDel = y.count - nKeep
      if !alreadyTrained && (nKeep < 2 || nDel < 2) {
        self.fail(reject, "TOO_FEW", "Need at least 2 keeps and 2 deletes to start this model")
        return
      }

      let metrics = self.bank.updateSelected(
        modelId: modelId,
        rows: rows,
        labels: y,
        seed: Int(Date().timeIntervalSince1970) % 10_000,
        hyper: hyper,
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
            "trainSize": rows.count,
            "testSize": 0,
            "learningRate": Double(lr),
            "done": false,
          ])
        }
      )

      do {
        try self.bank.saveSelected(modelId: modelId, newIds: sampleIds, metrics: metrics, trainSize: rows.count)
      } catch {
        self.fail(reject, "SAVE_FAILED", error.localizedDescription)
        return
      }

      let result: [String: Any] = [
        "epochsRan": metrics.epoch,
        "bestTestLoss": metrics.trainLoss,
        "bestTestAcc": metrics.trainAcc,
        "accuracy": metrics.trainAcc,
        "f1": metrics.f1,
        "cancelled": self.stopRequested,
        "trainSize": rows.count,
        "testSize": 0,
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
}
