import Foundation

protocol FeatureScorer {
  var modelId: String { get }
  mutating func updateBatch(rows: [[Float]], labels: [Float], seed: Int) -> TrainingBatchMetrics
  func predict(_ row: [Float]) -> Float
  func save(to dir: URL) throws
  mutating func load(from dir: URL) throws
  var hasModel: Bool { get }
}

struct TrainingBatchMetrics {
  var trainLoss: Float
  var trainAcc: Float
  var f1: Float
  var innerLoss: Float
  var epoch: Int
  var accepted: Bool
}

func makeMetrics(labels: [Float], probs: [Float], innerLoss: Float, epoch: Int, accepted: Bool) -> TrainingBatchMetrics {
  TrainingBatchMetrics(
    trainLoss: logLoss(labels: labels, probs: probs),
    trainAcc: accuracy(labels: labels, probs: probs),
    f1: f1Score(labels: labels, probs: probs),
    innerLoss: innerLoss,
    epoch: epoch,
    accepted: accepted
  )
}

// MARK: - Logistic regression with Hessian memory

struct LogisticHessianScorer: FeatureScorer {
  let modelId = "logistic-l2"
  private var theta = [Float](repeating: 0, count: FeatureSchema.dim + 1)
  private var hess = [Float](repeating: 0, count: (FeatureSchema.dim + 1) * (FeatureSchema.dim + 1))
  var scaler = GuardedScaler()
  private(set) var hasModel = false

  mutating func updateBatch(rows: [[Float]], labels: [Float], seed: Int) -> TrainingBatchMetrics {
    _ = seed
    let zRows = rows.map { scaler.transform($0) }
    let weights = balancedWeights(labels: labels)
    if !hasModel {
      scaler.fit(rows)
      let zFit = rows.map { scaler.transform($0) }
      theta = fitLogisticLBFGS(zFit, labels: labels, weights: weights, maxIter: 500)
      hess = logisticHessian(zFit, labels: labels, theta: theta, weights: weights, addL2Identity: true)
      hasModel = true
    } else {
      let zFit = zRows
      let updated = updateLogisticQuadratic(zFit, labels: labels, theta: theta, hess: hess, weights: weights)
      let curv = dataCurvature(zFit, labels: labels, theta: updated, weights: weights)
      hess = addMatrices(hess, curv)
      theta = updated
    }
    let probs = zRows.map { logisticProba($0, theta: theta) }
    return makeMetrics(labels: labels, probs: probs, innerLoss: 0, epoch: 1, accepted: true)
  }

  func predict(_ row: [Float]) -> Float {
    guard hasModel else { return 0.5 }
    return logisticProba(scaler.transform(row), theta: theta)
  }

  func save(to dir: URL) throws {
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    let payload: [String: Any] = [
      "theta": jsonDoubles(theta),
      "hess": jsonDoubles(hess),
      "scalerMean": jsonDoubles(scaler.mean),
      "scalerScale": jsonDoubles(scaler.scale),
    ]
    let data = try JSONSerialization.data(withJSONObject: payload)
    try data.write(to: dir.appendingPathComponent("state.json"))
  }

  mutating func load(from dir: URL) throws {
    let data = try Data(contentsOf: dir.appendingPathComponent("state.json"))
    guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else { return }
    let t = floatVector(json["theta"])
    let h = floatVector(json["hess"])
    let m = floatVector(json["scalerMean"])
    let s = floatVector(json["scalerScale"])
    guard t.count == theta.count, h.count == hess.count, m.count == FeatureSchema.dim, s.count == FeatureSchema.dim else { return }
    theta = t
    hess = h
    scaler.mean = m
    scaler.scale = s
    hasModel = true
  }
}

private func logisticProba(_ x: [Float], theta: [Float]) -> Float {
  var z = theta[theta.count - 1]
  for i in 0..<x.count { z += x[i] * theta[i] }
  return sigmoid(z)
}

private func fitLogisticLBFGS(_ x: [[Float]], labels: [Float], weights: [Float], maxIter: Int) -> [Float] {
  let d = FeatureSchema.dim
  var theta = [Float](repeating: 0, count: d + 1)
  let lr: Float = 0.1
  for _ in 0..<maxIter {
    var grad = [Float](repeating: 0, count: d + 1)
    for i in 0..<x.count {
      let p = logisticProba(x[i], theta: theta)
      let r = weights[i] * (p - labels[i])
      for j in 0..<d { grad[j] += r * x[i][j] }
      grad[d] += r
    }
    for j in 0..<d {
      grad[j] += theta[j]
      theta[j] -= lr * grad[j] / Float(x.count)
    }
    theta[d] -= lr * grad[d] / Float(x.count)
  }
  return theta
}

private func logisticHessian(
  _ x: [[Float]], labels: [Float], theta: [Float], weights: [Float], addL2Identity: Bool
) -> [Float] {
  let d = FeatureSchema.dim + 1
  var h = [Float](repeating: 0, count: d * d)
  for i in 0..<x.count {
    let p = logisticProba(x[i], theta: theta)
    let c = weights[i] * p * (1 - p)
    var aug = x[i] + [1]
    for a in 0..<d {
      for b in 0..<d {
        h[a * d + b] += c * aug[a] * aug[b]
      }
    }
  }
  if addL2Identity {
    for j in 0..<(d - 1) { h[j * d + j] += 1 }
  }
  return h
}

private func dataCurvature(_ x: [[Float]], labels: [Float], theta: [Float], weights: [Float]) -> [Float] {
  logisticHessian(x, labels: labels, theta: theta, weights: weights, addL2Identity: false)
}

private func addMatrices(_ a: [Float], _ b: [Float]) -> [Float] {
  zip(a, b).map(+)
}

private func updateLogisticQuadratic(
  _ x: [[Float]], labels: [Float], theta prev: [Float], hess: [Float], weights: [Float]
) -> [Float] {
  let d = prev.count
  var theta = prev
  let lr: Float = 0.05
  for _ in 0..<200 {
    var grad = [Float](repeating: 0, count: d)
    for i in 0..<x.count {
      let p = logisticProba(x[i], theta: theta)
      let r = weights[i] * (p - labels[i])
      var aug = x[i] + [1]
      for j in 0..<d { grad[j] += r * aug[j] }
    }
    let diff = zip(theta, prev).map(-)
    for j in 0..<d {
      var hGrad: Float = 0
      for k in 0..<d { hGrad += hess[j * d + k] * diff[k] }
      grad[j] += hGrad
    }
    var moved = false
    for j in 0..<d {
      let step = lr * grad[j] / Float(max(1, x.count))
      if abs(step) > 1e-8 { moved = true }
      theta[j] -= step
    }
    if !moved { break }
  }
  return theta
}

// MARK: - Gaussian Naive Bayes

struct GaussianNBScorer: FeatureScorer {
  let modelId = "gaussian-nb"
  var scaler = GuardedScaler()
  private var classCounts = [Float(0), Float(0)]
  private var mean0 = [Float](repeating: 0, count: FeatureSchema.dim)
  private var mean1 = [Float](repeating: 0, count: FeatureSchema.dim)
  private var var0 = [Float](repeating: 1, count: FeatureSchema.dim)
  private var var1 = [Float](repeating: 1, count: FeatureSchema.dim)
  private(set) var hasModel = false

  mutating func updateBatch(rows: [[Float]], labels: [Float], seed: Int) -> TrainingBatchMetrics {
    _ = seed
    if !hasModel { scaler.fit(rows); hasModel = true }
    let zRows = rows.map { scaler.transform($0) }
    let weights = balancedWeights(labels: labels)
    for i in 0..<zRows.count {
      let y = labels[i] >= 0.5 ? 1 : 0
      let w = weights[i]
      classCounts[y] += w
      var mean = y == 0 ? mean0 : mean1
      var v = y == 0 ? var0 : var1
      for j in 0..<FeatureSchema.dim {
        let delta = zRows[i][j] - mean[j]
        mean[j] += w * delta / max(classCounts[y], 1e-6)
        v[j] += w * delta * (zRows[i][j] - mean[j])
      }
      if y == 0 { mean0 = mean; var0 = v } else { mean1 = mean; var1 = v }
    }
    for j in 0..<FeatureSchema.dim {
      var0[j] = max(var0[j] / max(classCounts[0], 1), 1e-6)
      var1[j] = max(var1[j] / max(classCounts[1], 1), 1e-6)
    }
    let probs = zRows.map { predictScaled($0) }
    return makeMetrics(labels: labels, probs: probs, innerLoss: 0, epoch: 1, accepted: true)
  }

  private func predictScaled(_ z: [Float]) -> Float {
    let l0 = logPrior(0) + gaussLogLik(z, mean: mean0, var: var0)
    let l1 = logPrior(1) + gaussLogLik(z, mean: mean1, var: var1)
    let m = max(l0, l1)
    let e0 = exp(l0 - m)
    let e1 = exp(l1 - m)
    return e1 / (e0 + e1)
  }

  func predict(_ row: [Float]) -> Float {
    guard hasModel else { return 0.5 }
    return predictScaled(scaler.transform(row))
  }

  private func logPrior(_ c: Int) -> Float {
    let total = classCounts[0] + classCounts[1]
    guard total > 0 else { return 0 }
    return log(classCounts[c] / total)
  }

  private func gaussLogLik(_ x: [Float], mean: [Float], var v: [Float]) -> Float {
    var s: Float = 0
    for j in 0..<x.count {
      s -= 0.5 * log(2 * Float.pi * v[j])
      s -= 0.5 * (x[j] - mean[j]) * (x[j] - mean[j]) / v[j]
    }
    return s
  }

  func save(to dir: URL) throws {
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    let payload: [String: Any] = [
      "classCounts": jsonDoubles(classCounts),
      "mean0": jsonDoubles(mean0), "mean1": jsonDoubles(mean1),
      "var0": jsonDoubles(var0), "var1": jsonDoubles(var1),
      "scalerMean": jsonDoubles(scaler.mean), "scalerScale": jsonDoubles(scaler.scale),
    ]
    try JSONSerialization.data(withJSONObject: payload).write(to: dir.appendingPathComponent("state.json"))
  }

  mutating func load(from dir: URL) throws {
    let data = try Data(contentsOf: dir.appendingPathComponent("state.json"))
    guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else { return }
    let counts = floatVector(json["classCounts"])
    let m0 = floatVector(json["mean0"])
    let m1 = floatVector(json["mean1"])
    let v0 = floatVector(json["var0"])
    let v1 = floatVector(json["var1"])
    let sm = floatVector(json["scalerMean"])
    let ss = floatVector(json["scalerScale"])
    guard counts.count == 2,
          m0.count == FeatureSchema.dim, m1.count == FeatureSchema.dim,
          v0.count == FeatureSchema.dim, v1.count == FeatureSchema.dim,
          sm.count == FeatureSchema.dim, ss.count == FeatureSchema.dim else { return }
    classCounts = counts
    mean0 = m0
    mean1 = m1
    var0 = v0
    var1 = v1
    scaler.mean = sm
    scaler.scale = ss
    hasModel = true
  }
}
