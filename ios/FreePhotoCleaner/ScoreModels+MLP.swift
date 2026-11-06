import Foundation

// MARK: - MLP

struct MLPHyperparams {
  var batchSize: Int = 8
  var learningRate: Float = 0.001
  var scheduler: String = "constant"
  var stepGamma: Float = 0.5
  var stepEvery: Int = 5
  var expDecay: Float = 0.95
  var maxEpochs: Int = 80
}

struct CatBoostHyperparams {
  var learningRate: Float = 0.1
  var maxDepth: Int = 3
  var minLeaf: Int = 8
  var maxNewTrees: Int = 40
}

func scheduledLearningRate(_ hyper: MLPHyperparams, epoch: Int, maxEpoch: Int) -> Float {
  let base = hyper.learningRate
  switch hyper.scheduler {
  case "cosine":
    let t = Float(max(0, epoch - 1)) / Float(max(maxEpoch - 1, 1))
    return base * 0.5 * (1 + cos(Float.pi * t))
  case "step":
    let every = max(hyper.stepEvery, 1)
    let steps = epoch / every
    return base * pow(max(hyper.stepGamma, 0), Float(steps))
  case "exponential":
    return base * pow(max(hyper.expDecay, 0), Float(max(0, epoch - 1)))
  default:
    return base
  }
}

struct MLPScorer: FeatureScorer {
  let modelId: String
  private let hidden: [Int]
  var scaler = GuardedScaler()
  private var weights: [[[Float]]] = []
  private var biases: [[Float]] = []
  private(set) var hasModel = false

  init(modelId: String, hidden: [Int]) {
    self.modelId = modelId
    self.hidden = hidden
  }

  mutating func updateBatch(rows: [[Float]], labels: [Float], seed: Int) -> TrainingBatchMetrics {
    update(
      rows: rows,
      labels: labels,
      seed: seed,
      hyper: MLPHyperparams(),
      shouldStop: { false },
      onEpoch: { _, _, _, _, _, _ in }
    )
  }

  mutating func update(
    rows: [[Float]],
    labels: [Float],
    seed: Int,
    hyper: MLPHyperparams,
    shouldStop: () -> Bool,
    onEpoch: (Int, Float, Float, Float, Float, Float) -> Void
  ) -> TrainingBatchMetrics {
    if !hasModel {
      scaler.fit(rows)
      initWeights(seed: seed)
      hasModel = true
    }
    let split = stratifiedInnerSplit(labels: labels, seed: seed)
    let trainIdx: [Int]
    let valIdx: [Int]
    if let s = split {
      trainIdx = s.0
      valIdx = s.1
    } else {
      trainIdx = Array(0..<rows.count)
      valIdx = []
    }
    let zAll = rows.map { scaler.transform($0) }
    var bestLoss = Float.infinity
    var bestWeights = weights
    var bestBiases = biases
    var bestEpoch = 0
    var wait = 0
    let maxEpoch = max(1, hyper.maxEpochs)
    for epoch in 1...maxEpoch {
      if shouldStop() { break }
      let lr = scheduledLearningRate(hyper, epoch: epoch, maxEpoch: maxEpoch)
      trainEpoch(
        indices: trainIdx,
        zRows: zAll,
        labels: labels,
        lr: lr,
        batchSize: hyper.batchSize,
        shouldStop: shouldStop
      )
      let trainLoss = batchLogLoss(indices: trainIdx, zRows: zAll, labels: labels)
      let vLoss = valIdx.isEmpty ? trainLoss : batchLogLoss(indices: valIdx, zRows: zAll, labels: labels)
      let trainProbs = trainIdx.map { forward(zAll[$0]) }
      let trainLabs = trainIdx.map { labels[$0] }
      let trainAcc = accuracy(labels: trainLabs, probs: trainProbs)
      let valAcc: Float
      if valIdx.isEmpty {
        valAcc = trainAcc
      } else {
        valAcc = accuracy(labels: valIdx.map { labels[$0] }, probs: valIdx.map { forward(zAll[$0]) })
      }
      onEpoch(epoch, trainLoss, vLoss, trainAcc, valAcc, lr)
      if vLoss < bestLoss - 1e-6 {
        bestLoss = vLoss
        bestEpoch = epoch
        bestWeights = weights
        bestBiases = biases
        wait = 0
      } else {
        wait += 1
        if wait >= 8 { break }
      }
    }
    weights = bestWeights
    biases = bestBiases
    let probs = zAll.map { forward($0) }
    return makeMetrics(labels: labels, probs: probs, innerLoss: bestLoss, epoch: bestEpoch, accepted: bestEpoch > 0)
  }

  func predict(_ row: [Float]) -> Float {
    guard hasModel else { return 0.5 }
    return forward(scaler.transform(row))
  }

  private mutating func initWeights(seed: Int) {
    var rng = SeededRNG(seed: UInt64(seed))
    var layers = [FeatureSchema.dim] + hidden + [1]
    weights = []
    biases = []
    for l in 0..<(layers.count - 1) {
      let fanIn = layers[l]
      let fanOut = layers[l + 1]
      let limit = sqrt(6 / Float(fanIn + fanOut))
      var w = [[Float]]()
      for _ in 0..<fanOut {
        var row = [Float]()
        for _ in 0..<fanIn {
          row.append(Float.random(in: -limit...limit, using: &rng))
        }
        w.append(row)
      }
      weights.append(w)
      biases.append([Float](repeating: 0, count: fanOut))
    }
  }

  private mutating func trainEpoch(
    indices: [Int],
    zRows: [[Float]],
    labels: [Float],
    lr: Float,
    batchSize: Int,
    shouldStop: () -> Bool
  ) {
    let sampleWeights = balancedWeights(labels: labels)
    var order = indices
    var rng = SeededRNG(seed: UInt64(indices.count &* 17 &+ max(1, Int(lr * 10_000))))
    order.shuffle(using: &rng)
    let bs = max(1, batchSize)
    var start = 0
    while start < order.count {
      if shouldStop() { return }
      let end = min(start + bs, order.count)
      applyMiniBatch(Array(order[start..<end]), zRows: zRows, labels: labels, sampleWeights: sampleWeights, lr: lr)
      start = end
    }
  }

  private mutating func applyMiniBatch(
    _ batch: [Int],
    zRows: [[Float]],
    labels: [Float],
    sampleWeights: [Float],
    lr: Float
  ) {
    guard !batch.isEmpty, !weights.isEmpty else { return }
    var gradW = weights.map { layer in layer.map { row in [Float](repeating: 0, count: row.count) } }
    var gradB = biases.map { row in [Float](repeating: 0, count: row.count) }
    for idx in batch {
      var x = zRows[idx]
      var acts: [[Float]] = [x]
      for l in 0..<weights.count {
        x = matVec(weights[l], x, bias: biases[l], relu: l < weights.count - 1)
        acts.append(x)
      }
      guard let last = x.first else { continue }
      let y = labels[idx]
      let p = sigmoid(last)
      var delta = [sampleWeights[idx] * (p - y)]
      for l in stride(from: weights.count - 1, through: 0, by: -1) {
        let inp = acts[l]
        var nextDelta = [Float](repeating: 0, count: inp.count)
        for o in 0..<weights[l].count {
          let g = o < delta.count ? delta[o] : 0
          gradB[l][o] += g
          for i in 0..<inp.count {
            gradW[l][o][i] += g * inp[i]
            nextDelta[i] += g * weights[l][o][i]
          }
        }
        if l > 0 {
          for i in 0..<nextDelta.count {
            nextDelta[i] *= acts[l][i] > 0 ? 1 : 0
          }
          delta = nextDelta
        }
      }
    }
    let inv = lr / Float(batch.count)
    for l in 0..<weights.count {
      for o in 0..<weights[l].count {
        biases[l][o] -= inv * gradB[l][o]
        for i in 0..<weights[l][o].count {
          weights[l][o][i] -= inv * gradW[l][o][i]
        }
      }
    }
  }

  private func batchLogLoss(indices: [Int], zRows: [[Float]], labels: [Float]) -> Float {
    let probs = indices.map { forward(zRows[$0]) }
    let labs = indices.map { labels[$0] }
    return logLoss(labels: labs, probs: probs)
  }

  private func forward(_ x: [Float]) -> Float {
    var h = x
    for l in 0..<weights.count {
      h = matVec(weights[l], h, bias: biases[l], relu: l < weights.count - 1)
    }
    return sigmoid(h[0])
  }

  func save(to dir: URL) throws {
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    let payload: [String: Any] = [
      "weights": jsonDoubles3(weights),
      "biases": jsonDoubles2(biases),
      "scalerMean": jsonDoubles(scaler.mean),
      "scalerScale": jsonDoubles(scaler.scale),
      "hidden": hidden,
    ]
    try JSONSerialization.data(withJSONObject: payload).write(to: dir.appendingPathComponent("state.json"))
  }

  mutating func load(from dir: URL) throws {
    let data = try Data(contentsOf: dir.appendingPathComponent("state.json"))
    guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else { return }
    let w = floatTensor(json["weights"])
    let b = floatMatrix(json["biases"])
    let sm = floatVector(json["scalerMean"])
    let ss = floatVector(json["scalerScale"])
    let fanIn = [FeatureSchema.dim] + hidden
    guard w.count == hidden.count + 1, w.count == b.count,
          sm.count == FeatureSchema.dim, ss.count == FeatureSchema.dim else { return }
    for l in 0..<w.count {
      let fanOut = l == w.count - 1 ? 1 : hidden[l]
      guard w[l].count == fanOut, b[l].count == fanOut,
            w[l].allSatisfy({ $0.count == fanIn[l] }) else { return }
    }
    weights = w
    biases = b
    scaler.mean = sm
    scaler.scale = ss
    hasModel = true
  }
}

private func matVec(_ w: [[Float]], _ x: [Float], bias: [Float], relu: Bool) -> [Float] {
  var out = [Float](repeating: 0, count: w.count)
  for o in 0..<w.count {
    var s = bias[o]
    for i in 0..<x.count { s += w[o][i] * x[i] }
    out[o] = relu ? max(0, s) : s
  }
  return out
}

struct SeededRNG: RandomNumberGenerator {
  private var state: UInt64
  init(seed: UInt64) { state = seed == 0 ? 1 : seed }
  mutating func next() -> UInt64 {
    state = state &* 6364136223846793005 &+ 1
    return state
  }
}

extension Float {
  static func random(in range: ClosedRange<Float>, using generator: inout SeededRNG) -> Float {
    let u = Float(generator.next() % 10_000) / 10_000
    return range.lowerBound + u * (range.upperBound - range.lowerBound)
  }
}

// MARK: - Gradient boosted trees (CatBoost-style continuation)

struct BoostedTreeNode: Codable {
  var feature: Int = -1
  var threshold: Float = 0
  var left: Int = -1
  var right: Int = -1
  var value: Float = 0
  var isLeaf: Bool = true
}

struct BoostedTreeScorer: FeatureScorer {
  let modelId = "catboost"
  private var trees: [[BoostedTreeNode]] = []
  private var baseScore: Float = 0
  private(set) var hasModel = false
  private var learningRate: Float = 0.1
  private var trainHyper = CatBoostHyperparams()

  mutating func updateBatch(rows: [[Float]], labels: [Float], seed: Int) -> TrainingBatchMetrics {
    updateBatch(rows: rows, labels: labels, seed: seed, hyper: CatBoostHyperparams())
  }

  mutating func updateBatch(
    rows: [[Float]], labels: [Float], seed: Int, hyper: CatBoostHyperparams
  ) -> TrainingBatchMetrics {
    trainHyper = hyper
    learningRate = hyper.learningRate
    let weights = balancedWeights(labels: labels)
    var extra = hyper.maxNewTrees
    if stratifiedInnerSplit(labels: labels, seed: seed) != nil, hasModel {
      extra = min(20, hyper.maxNewTrees)
    }
    if extra <= 0 && hasModel {
      let probs = rows.map { predict($0) }
      return makeMetrics(labels: labels, probs: probs, innerLoss: 0, epoch: 0, accepted: false)
    }
    if extra <= 0 { extra = min(10, hyper.maxNewTrees) }

    var preds = rows.map { rawScore($0) }
    for _ in 0..<extra {
      var grad = [Float]()
      var hess = [Float]()
      for i in 0..<rows.count {
        let p = sigmoid(preds[i])
        grad.append(weights[i] * (p - labels[i]))
        hess.append(max(weights[i] * p * (1 - p), 1e-6))
      }
      let tree = buildTree(rows: rows, grad: grad, hess: hess, depth: 0, seed: seed)
      trees.append(tree)
      for i in 0..<rows.count {
        preds[i] += learningRate * evalTree(tree, rows[i])
      }
    }
    hasModel = !trees.isEmpty
    let probs = preds.map(sigmoid)
    return makeMetrics(labels: labels, probs: probs, innerLoss: 0, epoch: extra, accepted: extra > 0)
  }

  private func rawScore(_ row: [Float]) -> Float {
    baseScore + learningRate * evalForest(row)
  }

  private func evalForest(_ row: [Float]) -> Float {
    trees.reduce(0) { $0 + evalTree($1, row) }
  }

  private func evalTree(_ tree: [BoostedTreeNode], _ row: [Float]) -> Float {
    guard !tree.isEmpty else { return 0 }
    var idx = 0
    var steps = 0
    while idx >= 0, idx < tree.count, !tree[idx].isLeaf, steps < 64 {
      let n = tree[idx]
      let goLeft = n.feature >= 0 && n.feature < row.count && row[n.feature] <= n.threshold
      idx = goLeft ? n.left : n.right
      steps += 1
    }
    guard idx >= 0, idx < tree.count else { return 0 }
    return tree[idx].value
  }

  private func buildTree(rows: [[Float]], grad: [Float], hess: [Float], depth: Int, seed: Int) -> [BoostedTreeNode] {
    var nodes: [BoostedTreeNode] = []
    func leaf(_ g: Float, _ h: Float) -> Int {
      let v = -g / (h + 1e-6)
      let idx = nodes.count
      nodes.append(BoostedTreeNode(value: v, isLeaf: true))
      return idx
    }
    func build(indices: [Int], d: Int) -> Int {
      let gSum = indices.reduce(Float(0)) { $0 + grad[$1] }
      let hSum = indices.reduce(Float(0)) { $0 + hess[$1] }
      if d >= trainHyper.maxDepth || indices.count < trainHyper.minLeaf * 2 {
        return leaf(gSum, hSum)
      }
      var bestGain: Float = -1
      var bestF = 0
      var bestT: Float = 0
      var bestL: [Int] = []
      var bestR: [Int] = []
      for f in 0..<FeatureSchema.dim {
        let vals = indices.map { rows[$0][f] }.sorted()
        for t in stride(from: 0, to: vals.count - 1, by: max(1, vals.count / 8)) {
          let thr = (vals[t] + vals[t + 1]) * 0.5
          let l = indices.filter { rows[$0][f] <= thr }
          let r = indices.filter { rows[$0][f] > thr }
          if l.count < trainHyper.minLeaf || r.count < trainHyper.minLeaf { continue }
          let gl = l.reduce(0) { $0 + grad[$1] }
          let hl = l.reduce(0) { $0 + hess[$1] }
          let gr = r.reduce(0) { $0 + grad[$1] }
          let hr = r.reduce(0) { $0 + hess[$1] }
          let gain = gl * gl / (hl + 1e-6) + gr * gr / (hr + 1e-6) - gSum * gSum / (hSum + 1e-6)
          if gain > bestGain {
            bestGain = gain
            bestF = f
            bestT = thr
            bestL = l
            bestR = r
          }
        }
      }
      if bestGain <= 0 {
        return leaf(gSum, hSum)
      }
      let nodeIdx = nodes.count
      nodes.append(BoostedTreeNode(feature: bestF, threshold: bestT, isLeaf: false))
      let left = build(indices: bestL, d: d + 1)
      let right = build(indices: bestR, d: d + 1)
      nodes[nodeIdx].left = left
      nodes[nodeIdx].right = right
      return nodeIdx
    }
    _ = build(indices: Array(0..<rows.count), d: depth)
    return nodes
  }

  func predict(_ row: [Float]) -> Float {
    guard hasModel else { return 0.5 }
    return sigmoid(baseScore + evalForest(row) * learningRate)
  }

  func save(to dir: URL) throws {
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    let enc = JSONEncoder()
    try enc.encode(trees).write(to: dir.appendingPathComponent("trees.bin"))
    let meta: [String: Any] = ["baseScore": Double(baseScore.isFinite ? baseScore : 0)]
    try JSONSerialization.data(withJSONObject: meta).write(to: dir.appendingPathComponent("meta.json"))
  }

  mutating func load(from dir: URL) throws {
    let data = try Data(contentsOf: dir.appendingPathComponent("trees.bin"))
    trees = try JSONDecoder().decode([[BoostedTreeNode]].self, from: data)
    if let m = try JSONSerialization.jsonObject(with: Data(contentsOf: dir.appendingPathComponent("meta.json"))) as? [String: Any] {
      baseScore = (m["baseScore"] as? NSNumber)?.floatValue ?? 0
    }
    hasModel = !trees.isEmpty
  }
}

extension Array {
  mutating func shuffle(using generator: inout SeededRNG) {
    guard count > 1 else { return }
    for i in stride(from: count - 1, through: 1, by: -1) {
      let j = Int(generator.next() % UInt64(i + 1))
      swapAt(i, j)
    }
  }
}

func stratifiedInnerSplit(labels: [Float], seed: Int) -> ([Int], [Int])? {
  let n = labels.count
  if n < 10 { return nil }
  var c0: [Int] = []
  var c1: [Int] = []
  for i in 0..<n {
    if labels[i] >= 0.5 { c1.append(i) } else { c0.append(i) }
  }
  if c0.count < 2 || c1.count < 2 { return nil }
  let valN0 = max(1, c0.count / 5)
  let valN1 = max(1, c1.count / 5)
  var rng = SeededRNG(seed: UInt64(seed))
  c0.shuffle(using: &rng)
  c1.shuffle(using: &rng)
  let val = Array(c0.prefix(valN0)) + Array(c1.prefix(valN1))
  let valSet = Set(val)
  let train = (0..<n).filter { !valSet.contains($0) }
  return (train, val)
}

// MARK: - Trained-photo bitsets

/// One shared slot table (id → bit index) plus a bit vector per model.
/// A set bit means that photo's label was already added to that model.
final class TrainedPhotoIndex {
  private let root: URL
  private var ids: [String] = []
  private var slotById: [String: Int] = [:]
  private var bits: [String: [UInt8]] = [:]

  init(root: URL) {
    self.root = root
    loadSlots()
  }

  func contains(_ localId: String, modelId: String) -> Bool {
    guard let slot = slotById[localId] else { return false }
    return isSet(bits(for: modelId), slot)
  }

  func count(modelId: String) -> Int {
    bits(for: modelId).reduce(0) { $0 + $1.nonzeroBitCount }
  }

  func mark(_ localIds: [String], modelId: String) {
    var buf = bits(for: modelId)
    var slotsDirty = false
    for id in localIds {
      let slot: Int
      if let existing = slotById[id] {
        slot = existing
      } else {
        slot = ids.count
        ids.append(id)
        slotById[id] = slot
        slotsDirty = true
      }
      setBit(&buf, slot)
    }
    bits[modelId] = buf
    if slotsDirty { saveSlots() }
    saveBits(modelId, buf)
  }

  func clear(modelId: String) {
    bits[modelId] = []
    try? FileManager.default.removeItem(at: bitsURL(modelId))
  }

  func resetAll() {
    ids = []
    slotById = [:]
    bits = [:]
  }

  private func bits(for modelId: String) -> [UInt8] {
    if let cached = bits[modelId] { return cached }
    let loaded = (try? Data(contentsOf: bitsURL(modelId))).map { [UInt8]($0) } ?? []
    bits[modelId] = loaded
    return loaded
  }

  private func bitsURL(_ modelId: String) -> URL {
    root.appendingPathComponent(modelId, isDirectory: true).appendingPathComponent("trained.bits")
  }

  private func slotsURL() -> URL {
    root.appendingPathComponent("asset_slots.bin")
  }

  private func isSet(_ buf: [UInt8], _ slot: Int) -> Bool {
    let i = slot >> 3
    guard i >= 0, i < buf.count else { return false }
    return buf[i] & UInt8(1 << (slot & 7)) != 0
  }

  private func setBit(_ buf: inout [UInt8], _ slot: Int) {
    let i = slot >> 3
    if buf.count <= i {
      buf.append(contentsOf: repeatElement(0, count: i + 1 - buf.count))
    }
    buf[i] |= UInt8(1 << (slot & 7))
  }

  private func loadSlots() {
    guard let data = try? Data(contentsOf: slotsURL()), data.count >= 4 else { return }
    var offset = 0
    let count = Int(readUInt32(data, &offset))
    ids.reserveCapacity(count)
    for _ in 0..<count {
      guard offset + 4 <= data.count else { break }
      let len = Int(readUInt32(data, &offset))
      guard len >= 0, offset + len <= data.count else { break }
      let slice = data.subdata(in: offset..<(offset + len))
      offset += len
      guard let id = String(data: slice, encoding: .utf8) else { continue }
      slotById[id] = ids.count
      ids.append(id)
    }
  }

  private func saveSlots() {
    var data = Data()
    appendUInt32(&data, UInt32(ids.count))
    for id in ids {
      let bytes = Data(id.utf8)
      appendUInt32(&data, UInt32(bytes.count))
      data.append(bytes)
    }
    try? data.write(to: slotsURL(), options: .atomic)
  }

  private func saveBits(_ modelId: String, _ buf: [UInt8]) {
    let dir = root.appendingPathComponent(modelId, isDirectory: true)
    try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    try? Data(buf).write(to: bitsURL(modelId), options: .atomic)
  }

  private func readUInt32(_ data: Data, _ offset: inout Int) -> UInt32 {
    let b0 = UInt32(data[offset])
    let b1 = UInt32(data[offset + 1])
    let b2 = UInt32(data[offset + 2])
    let b3 = UInt32(data[offset + 3])
    offset += 4
    return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
  }

  private func appendUInt32(_ data: inout Data, _ value: UInt32) {
    var le = value.littleEndian
    withUnsafeBytes(of: &le) { data.append(contentsOf: $0) }
  }
}

// MARK: - Bank of all scorers

struct ModelScoreRecord {
  var trainedAtMs: Double = 0
  var accuracy: Float = 0
  var f1: Float = 0
  var trainSize: Int = 0
}

final class FeatureModelBank {
  private var logistic = LogisticHessianScorer()
  private var catboost = BoostedTreeScorer()
  private var mlpSmall = MLPScorer(modelId: "mlp-small", hidden: [16])
  private var mlpBig = MLPScorer(modelId: "mlp-big", hidden: [128, 64, 32, 16, 8])
  private var naiveBayes = GaussianNBScorer()
  private var records: [String: ModelScoreRecord] = [:]
  private var trainedIndex: TrainedPhotoIndex!
  var activeModelId = "catboost"

  private var modelsDir: URL {
    let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    let dir = docs.appendingPathComponent("ai_feature_models/v\(FeatureSchema.version)", isDirectory: true)
    try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    return dir
  }

  private var activeURL: URL { modelsDir.appendingPathComponent("active_model_id.txt") }

  init() {
    let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    let dir = docs.appendingPathComponent("ai_feature_models/v\(FeatureSchema.version)", isDirectory: true)
    try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    trainedIndex = TrainedPhotoIndex(root: dir)
    purgeLegacyCNNCheckpoints()
    migrateJointIncorporatedFile()
    loadAll()
  }

  func purgeLegacyCNNCheckpoints() {
    let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    let legacy = docs.appendingPathComponent("ai_models", isDirectory: true)
    guard let contents = try? FileManager.default.contentsOfDirectory(at: legacy, includingPropertiesForKeys: nil) else { return }
    for url in contents {
      let meta = url.appendingPathComponent("meta.json")
      guard let data = try? Data(contentsOf: meta),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
        try? FileManager.default.removeItem(at: url)
        continue
      }
      if json["featureSchemaVersion"] as? Int != FeatureSchema.version {
        try? FileManager.default.removeItem(at: url)
      }
    }
  }

  /// Older builds trained every model at once and stored one shared id list.
  /// Keep that list only for the model the user had selected, and drop the other
  /// weight files so each model can be trained on the same photos on purpose.
  private func migrateJointIncorporatedFile() {
    let url = modelsDir.appendingPathComponent("incorporated.json")
    guard let data = try? Data(contentsOf: url),
          let arr = try? JSONSerialization.jsonObject(with: data) as? [String] else { return }
    let active = (try? String(contentsOf: activeURL, encoding: .utf8))?
      .trimmingCharacters(in: .whitespacesAndNewlines) ?? "logistic-l2"
    let keep = FeatureSchema.scoreModelIds.contains(active) ? active : "logistic-l2"
    if modelHasState(keep) {
      trainedIndex.mark(arr, modelId: keep)
    }
    for id in FeatureSchema.scoreModelIds where id != keep {
      try? FileManager.default.removeItem(at: modelDir(id))
      trainedIndex.clear(modelId: id)
    }
    try? FileManager.default.removeItem(at: url)
  }

  private func modelHasState(_ id: String) -> Bool {
    let dir = modelDir(id)
    let fm = FileManager.default
    return fm.fileExists(atPath: dir.appendingPathComponent("state.json").path)
      || fm.fileExists(atPath: dir.appendingPathComponent("trees.bin").path)
  }

  func loadAll() {
    if let id = (try? String(contentsOf: activeURL, encoding: .utf8))?.trimmingCharacters(in: .whitespacesAndNewlines),
       FeatureSchema.scoreModelIds.contains(id) {
      activeModelId = id
    }
    try? logistic.load(from: modelDir("logistic-l2"))
    try? catboost.load(from: modelDir("catboost"))
    try? mlpSmall.load(from: modelDir("mlp-small"))
    try? mlpBig.load(from: modelDir("mlp-big"))
    if !mlpBig.hasModel,
       FileManager.default.fileExists(atPath: modelDir("mlp-big").appendingPathComponent("state.json").path) {
      try? FileManager.default.removeItem(at: modelDir("mlp-big"))
      trainedIndex.clear(modelId: "mlp-big")
    }
    try? naiveBayes.load(from: modelDir("gaussian-nb"))
    for id in FeatureSchema.scoreModelIds {
      records[id] = loadRecord(id)
    }
  }

  func isIncorporated(_ localId: String, modelId: String) -> Bool {
    trainedIndex.contains(localId, modelId: modelId)
  }

  func hasModel(_ modelId: String) -> Bool {
    switch modelId {
    case "logistic-l2": return logistic.hasModel
    case "catboost": return catboost.hasModel
    case "mlp-small": return mlpSmall.hasModel
    case "mlp-big": return mlpBig.hasModel
    case "gaussian-nb": return naiveBayes.hasModel
    default: return false
    }
  }

  func saveSelected(modelId: String, newIds: [String], metrics: TrainingBatchMetrics, trainSize: Int) throws {
    trainedIndex.mark(newIds, modelId: modelId)
    let record = ModelScoreRecord(
      trainedAtMs: Date().timeIntervalSince1970 * 1000,
      accuracy: metrics.trainAcc,
      f1: metrics.f1,
      trainSize: trainSize
    )
    records[modelId] = record
    let dir = modelDir(modelId)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    let meta: [String: Any] = [
      "trainedAt": record.trainedAtMs,
      "accuracy": Double(record.accuracy),
      "f1": Double(record.f1),
      "trainSize": trainSize,
    ]
    try JSONSerialization.data(withJSONObject: meta).write(to: dir.appendingPathComponent("metrics.json"))
    switch modelId {
    case "logistic-l2": try logistic.save(to: dir)
    case "catboost": try catboost.save(to: dir)
    case "mlp-small": try mlpSmall.save(to: dir)
    case "mlp-big": try mlpBig.save(to: dir)
    case "gaussian-nb": try naiveBayes.save(to: dir)
    default: break
    }
    try? activeModelId.write(to: activeURL, atomically: true, encoding: .utf8)
  }

  func deleteModel(_ modelId: String) {
    trainedIndex.clear(modelId: modelId)
    records[modelId] = nil
    try? FileManager.default.removeItem(at: modelDir(modelId))
    switch modelId {
    case "logistic-l2": logistic = LogisticHessianScorer()
    case "catboost": catboost = BoostedTreeScorer()
    case "mlp-small": mlpSmall = MLPScorer(modelId: "mlp-small", hidden: [16])
    case "mlp-big": mlpBig = MLPScorer(modelId: "mlp-big", hidden: [128, 64, 32, 16, 8])
    case "gaussian-nb": naiveBayes = GaussianNBScorer()
    default: break
    }
  }

  func resetAll() {
    try? FileManager.default.removeItem(at: modelsDir)
    trainedIndex.resetAll()
    trainedIndex = TrainedPhotoIndex(root: modelsDir)
    records = [:]
    logistic = LogisticHessianScorer()
    catboost = BoostedTreeScorer()
    mlpSmall = MLPScorer(modelId: "mlp-small", hidden: [16])
    mlpBig = MLPScorer(modelId: "mlp-big", hidden: [128, 64, 32, 16, 8])
    naiveBayes = GaussianNBScorer()
  }

  private func loadRecord(_ id: String) -> ModelScoreRecord? {
    let url = modelDir(id).appendingPathComponent("metrics.json")
    guard let data = try? Data(contentsOf: url),
          let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
    return ModelScoreRecord(
      trainedAtMs: (json["trainedAt"] as? NSNumber)?.doubleValue ?? 0,
      accuracy: (json["accuracy"] as? NSNumber)?.floatValue ?? 0,
      f1: (json["f1"] as? NSNumber)?.floatValue ?? 0,
      trainSize: (json["trainSize"] as? NSNumber)?.intValue ?? 0
    )
  }

  func status(for modelId: String) -> [String: Any] {
    let rec = records[modelId]
    return [
      "modelId": modelId,
      "hasModel": hasModel(modelId),
      "trainedAt": rec?.trainedAtMs ?? 0,
      "accuracy": Double(rec?.accuracy ?? 0),
      "f1": Double(rec?.f1 ?? 0),
      "trainSize": rec?.trainSize ?? 0,
      "learnedCount": trainedIndex.count(modelId: modelId),
    ]
  }

  private func modelDir(_ id: String) -> URL {
    modelsDir.appendingPathComponent(id, isDirectory: true)
  }

  func hasAnyModel() -> Bool {
    logistic.hasModel || catboost.hasModel || mlpSmall.hasModel || mlpBig.hasModel || naiveBayes.hasModel
  }

  func predict(_ row: [Float], modelId: String) -> Float {
    switch modelId {
    case "logistic-l2": return logistic.predict(row)
    case "catboost": return catboost.predict(row)
    case "mlp-small": return mlpSmall.predict(row)
    case "mlp-big": return mlpBig.predict(row)
    case "gaussian-nb": return naiveBayes.predict(row)
    default: return logistic.predict(row)
    }
  }

  func updateSelected(
    modelId: String,
    rows: [[Float]],
    labels: [Float],
    seed: Int,
    mlpHyper: MLPHyperparams,
    logisticHyper: LogisticHyperparams,
    catboostHyper: CatBoostHyperparams,
    shouldStop: () -> Bool,
    onEpoch: (Int, Float, Float, Float, Float, Float) -> Void
  ) -> TrainingBatchMetrics {
    switch modelId {
    case "catboost":
      return catboost.updateBatch(rows: rows, labels: labels, seed: seed, hyper: catboostHyper)
    case "mlp-small":
      return mlpSmall.update(rows: rows, labels: labels, seed: seed, hyper: mlpHyper, shouldStop: shouldStop, onEpoch: onEpoch)
    case "mlp-big":
      return mlpBig.update(rows: rows, labels: labels, seed: seed, hyper: mlpHyper, shouldStop: shouldStop, onEpoch: onEpoch)
    case "gaussian-nb":
      return naiveBayes.updateBatch(rows: rows, labels: labels, seed: seed)
    default:
      return logistic.updateBatch(rows: rows, labels: labels, seed: seed, hyper: logisticHyper)
    }
  }

  func setActive(_ modelId: String) {
    activeModelId = modelId
    try? activeModelId.write(to: activeURL, atomically: true, encoding: .utf8)
  }

  func checkpointMeta(for modelId: String) -> [String: Any] {
    let rec = records[modelId]
    return [
      "modelId": modelId,
      "featureSchemaVersion": FeatureSchema.version,
      "trainedAt": rec?.trainedAtMs ?? 0,
      "label": FeatureSchema.label(for: modelId),
      "accuracy": Double(rec?.accuracy ?? 0),
      "f1": Double(rec?.f1 ?? 0),
      "bestTestAcc": Double(rec?.accuracy ?? 0),
      "trainSize": rec?.trainSize ?? 0,
      "learnedCount": trainedIndex.count(modelId: modelId),
    ]
  }

  func listCheckpoints() -> [[String: Any]] {
    FeatureSchema.scoreModelIds.compactMap { id in
      guard hasModel(id) else { return nil }
      return checkpointMeta(for: id)
    }
  }
}
