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

struct LogisticHyperparams {
  var C: Float = 1
  var maxIter: Int = 500
}

struct LogisticHessianScorer: FeatureScorer {
  let modelId = "logistic-l2"
  private var theta = [Float](repeating: 0, count: FeatureSchema.dim + 1)
  private var hess = [Float](repeating: 0, count: (FeatureSchema.dim + 1) * (FeatureSchema.dim + 1))
  var scaler = GuardedScaler()
  private(set) var hasModel = false
  private var lastHyper = LogisticHyperparams()

  mutating func updateBatch(rows: [[Float]], labels: [Float], seed: Int) -> TrainingBatchMetrics {
    updateBatch(rows: rows, labels: labels, seed: seed, hyper: LogisticHyperparams())
  }

  mutating func updateBatch(
    rows: [[Float]], labels: [Float], seed: Int, hyper: LogisticHyperparams
  ) -> TrainingBatchMetrics {
    _ = seed
    lastHyper = hyper
    let weights = balancedWeights(labels: labels)
    if !hasModel {
      scaler.fit(rows)
      let zFit = rows.map { scaler.transform($0) }
      theta = fitLogisticLBFGS(
        zFit, labels: labels, sampleWeights: weights, C: hyper.C, maxIter: hyper.maxIter, start: theta
      )
      hess = logisticHessian(
        zFit, labels: labels, theta: theta, weights: weights, C: hyper.C, addL2Identity: true
      )
      hasModel = true
    } else {
      let zFit = rows.map { scaler.transform($0) }
      let updated = updateLogisticQuadratic(
        zFit, labels: labels, theta: theta, hess: hess, weights: weights, C: hyper.C, maxIter: hyper.maxIter
      )
      let curv = dataCurvature(zFit, labels: labels, theta: updated, weights: weights, C: hyper.C)
      hess = addMatrices(hess, curv)
      theta = updated
    }
    let zEval = rows.map { scaler.transform($0) }
    let probs = zEval.map { logisticProba($0, theta: theta) }
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

private func stableLogLoss(z: Float, y: Float) -> Float {
  if z >= 0 { return log1p(exp(-z)) - y * z }
  return -y * z + log1p(exp(z))
}

private func logisticBatchObjectiveGrad(
  _ x: [[Float]], labels: [Float], sampleWeights: [Float], theta: [Float], C: Float
) -> (Float, [Float]) {
  let d = FeatureSchema.dim
  var grad = [Float](repeating: 0, count: d + 1)
  var loss: Float = 0
  for i in 0..<x.count {
    var z = theta[d]
    for j in 0..<d { z += x[i][j] * theta[j] }
    loss += sampleWeights[i] * stableLogLoss(z: z, y: labels[i])
    let p = sigmoid(z)
    let r = sampleWeights[i] * (p - labels[i])
    for j in 0..<d { grad[j] += r * x[i][j] }
    grad[d] += r
  }
  loss *= C
  for j in 0..<d {
    grad[j] = C * grad[j] + theta[j]
  }
  grad[d] *= C
  var l2: Float = 0
  for j in 0..<d { l2 += theta[j] * theta[j] }
  loss += 0.5 * l2
  return (loss, grad)
}

private func matVecSquare(_ m: [Float], _ v: [Float], n: Int) -> [Float] {
  var out = [Float](repeating: 0, count: n)
  for i in 0..<n {
    var s: Float = 0
    for k in 0..<n { s += m[i * n + k] * v[k] }
    out[i] = s
  }
  return out
}

private func dot(_ a: [Float], _ b: [Float]) -> Float {
  zip(a, b).reduce(Float(0)) { $0 + $1.0 * $1.1 }
}

private func lbfgsMinimize(
  start: [Float],
  maxIter: Int,
  gradTol: Float,
  valueAndGrad: ([Float]) -> (Float, [Float])
) -> [Float] {
  let n = start.count
  var x = start
  var (fx, g) = valueAndGrad(x)
  var sHist = [[Float]]()
  var yHist = [[Float]]()
  var rhoHist = [Float]()
  let mem = 10

  for _ in 0..<maxIter {
    if sqrt(dot(g, g)) < gradTol { break }
    let k = sHist.count
    var q = g
    var alpha = [Float](repeating: 0, count: mem)
    for i in stride(from: k - 1, through: 0, by: -1) {
      let idx = i % mem
      guard idx < rhoHist.count else { continue }
      alpha[idx] = rhoHist[idx] * dot(sHist[idx], q)
      for j in 0..<n { q[j] -= alpha[idx] * yHist[idx][j] }
    }
    var gamma: Float = 1
    if k > 0 {
      let last = (k - 1) % mem
      let yy = dot(yHist[last], yHist[last])
      if yy > 1e-12 { gamma = dot(sHist[last], yHist[last]) / yy }
    }
    var r = q.map { $0 * gamma }
    for i in 0..<k {
      let idx = i % mem
      guard idx < rhoHist.count else { continue }
      let beta = rhoHist[idx] * dot(yHist[idx], r)
      for j in 0..<n { r[j] += sHist[idx][j] * (alpha[idx] - beta) }
    }
    let direction = r.map { -$0 }
    var stepSize: Float = 1
    var accepted = x
    var acceptedG = g
    var acceptedF = fx
    for _ in 0..<25 {
      let trial = zip(x, direction).map { $0 + stepSize * $1 }
      let (fTrial, gTrial) = valueAndGrad(trial)
      if fTrial <= fx - 1e-4 * stepSize * dot(g, direction) {
        accepted = trial
        acceptedG = gTrial
        acceptedF = fTrial
        break
      }
      stepSize *= 0.5
    }
    if stepSize < 1e-10 { break }
    let s = zip(accepted, x).map { $0 - $1 }
    let y = zip(acceptedG, g).map { $0 - $1 }
    let sy = dot(s, y)
    if sy > 1e-10 {
      if sHist.count == mem {
        sHist.removeFirst()
        yHist.removeFirst()
        rhoHist.removeFirst()
      }
      sHist.append(s)
      yHist.append(y)
      rhoHist.append(1 / sy)
    }
    x = accepted
    g = acceptedG
    fx = acceptedF
  }
  return x
}

private func fitLogisticLBFGS(
  _ x: [[Float]], labels: [Float], sampleWeights: [Float], C: Float, maxIter: Int, start: [Float]
) -> [Float] {
  lbfgsMinimize(start: start, maxIter: maxIter, gradTol: 1e-6) { theta in
    logisticBatchObjectiveGrad(x, labels: labels, sampleWeights: sampleWeights, theta: theta, C: C)
  }
}

private func logisticHessian(
  _ x: [[Float]], labels: [Float], theta: [Float], weights: [Float], C: Float, addL2Identity: Bool
) -> [Float] {
  let d = FeatureSchema.dim + 1
  var h = [Float](repeating: 0, count: d * d)
  for i in 0..<x.count {
    let p = logisticProba(x[i], theta: theta)
    let c = C * weights[i] * p * (1 - p)
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

private func dataCurvature(
  _ x: [[Float]], labels: [Float], theta: [Float], weights: [Float], C: Float
) -> [Float] {
  logisticHessian(x, labels: labels, theta: theta, weights: weights, C: C, addL2Identity: false)
}

private func addMatrices(_ a: [Float], _ b: [Float]) -> [Float] {
  zip(a, b).map(+)
}

private func updateObjective(
  theta: [Float], prev: [Float], x: [[Float]], labels: [Float], weights: [Float], C: Float, hess: [Float]
) -> Float {
  let (batchLoss, _) = logisticBatchObjectiveGrad(x, labels: labels, sampleWeights: weights, theta: theta, C: C)
  let diff = zip(theta, prev).map(-)
  let hd = matVecSquare(hess, diff, n: theta.count)
  return batchLoss + 0.5 * dot(diff, hd)
}

private func updateObjectiveGrad(
  theta: [Float], prev: [Float], x: [[Float]], labels: [Float], weights: [Float], C: Float, hess: [Float]
) -> (Float, [Float]) {
  let (loss, gBatch) = logisticBatchObjectiveGrad(x, labels: labels, sampleWeights: weights, theta: theta, C: C)
  let diff = zip(theta, prev).map(-)
  let hGrad = matVecSquare(hess, diff, n: theta.count)
  var grad = gBatch
  for j in 0..<grad.count { grad[j] += hGrad[j] }
  return (loss + 0.5 * dot(diff, hGrad), grad)
}

private func updateLogisticQuadratic(
  _ x: [[Float]], labels: [Float], theta prev: [Float], hess: [Float], weights: [Float], C: Float, maxIter: Int
) -> [Float] {
  let startObj = updateObjective(theta: prev, prev: prev, x: x, labels: labels, weights: weights, C: C, hess: hess)
  let result = lbfgsMinimize(start: prev, maxIter: min(maxIter, 200), gradTol: 1e-5) { theta in
    updateObjectiveGrad(theta: theta, prev: prev, x: x, labels: labels, weights: weights, C: C, hess: hess)
  }
  let endObj = updateObjective(theta: result, prev: prev, x: x, labels: labels, weights: weights, C: C, hess: hess)
  if endObj > startObj + 1e-4 { return prev }
  return result
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
