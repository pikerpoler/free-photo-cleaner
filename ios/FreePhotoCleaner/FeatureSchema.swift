import Foundation

enum FeatureSchema {
  static let version = 1
  static let dim = 53

  static let photometricCount = 16
  static let geometryCount = 6
  static let categoryCount = 11
  static let visionCount = 6
  static let metaCount = 3

  static let confThreshold: Float = 0.25
  static let maxBoxes = 50
  static let faceConfThreshold: Float = 0.5
  static let decodeLongEdge = 640
  static let photometricLongEdge = 160
  static let yoloLetterbox = 416

  static let categoryNames: [String] = [
    "person", "vehicle", "outdoor", "animal", "accessory", "sports",
    "kitchen", "furniture", "electronic", "appliance", "indoor",
  ]

  /// COCO class index (0..79) -> category index (0..10), or -1 if unmapped.
  static let cocoToCategory: [Int16] = {
    var map = [Int16](repeating: -1, count: 80)
    func set(_ classes: [Int], _ cat: Int) {
      for c in classes { if c >= 0 && c < 80 { map[c] = Int16(cat) } }
    }
    set([0], 0)
    set([1, 2, 3, 4, 5, 6, 7, 8], 1)
    set([9, 10, 11, 12, 13], 2)
    set([14, 15, 16, 17, 18, 19, 20, 21, 22, 23], 3)
    set([24, 25, 26, 27, 28], 4)
    set([29, 30, 31, 32, 33, 34, 35, 36, 37, 38], 5)
    set([39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55], 6)
    set([56, 57, 59], 7)
    set([62, 63, 64, 65, 66, 67], 8)
    set([68, 69, 70, 71, 72, 78], 9)
    set([58, 60, 61, 73, 74, 75, 76, 77, 79], 10)
    return map
  }()

  static let scoreModelIds: [String] = [
    "logistic-l2",
    "catboost",
    "gaussian-nb",
    "mlp-small",
    "mlp-big",
  ]

  static func label(for modelId: String) -> String {
    switch modelId {
    case "logistic-l2": return "Logistic Regression"
    case "catboost": return "CatBoost"
    case "gaussian-nb": return "Gaussian Naive Bayes"
    case "mlp-small": return "small MLP"
    case "mlp-big": return "big MLP"
    default: return modelId
    }
  }
}

struct DetectionBox {
  var x1: Float
  var y1: Float
  var x2: Float
  var y2: Float
  var conf: Float
  var classId: Int
}

struct GuardedScaler: Codable {
  var mean: [Float]
  var scale: [Float]

  init() {
    mean = [Float](repeating: 0, count: FeatureSchema.dim)
    scale = [Float](repeating: 1, count: FeatureSchema.dim)
  }

  mutating func fit(_ rows: [[Float]]) {
    guard !rows.isEmpty else { return }
    let d = FeatureSchema.dim
    var mu = [Float](repeating: 0, count: d)
    for row in rows {
      for i in 0..<d { mu[i] += row[i] }
    }
    for i in 0..<d { mu[i] /= Float(rows.count) }
    var varSum = [Float](repeating: 0, count: d)
    for row in rows {
      for i in 0..<d {
        let dlt = row[i] - mu[i]
        varSum[i] += dlt * dlt
      }
    }
    mean = mu
    scale = (0..<d).map { i in
      let sd = sqrt(varSum[i] / Float(rows.count))
      return sd < 1e-8 ? 1 : sd
    }
  }

  func transform(_ row: [Float]) -> [Float] {
    (0..<FeatureSchema.dim).map { i in (row[i] - mean[i]) / scale[i] }
  }
}

func balancedWeights(labels: [Float]) -> [Float] {
  let n = labels.count
  guard n > 0 else { return [] }
  var nPos: Float = 0
  for y in labels where y >= 0.5 { nPos += 1 }
  let nNeg = Float(n) - nPos
  let nf = Float(n)
  let wp = nf / max(2 * max(nPos, 1), 1)
  let wn = nf / max(2 * max(nNeg, 1), 1)
  return labels.map { $0 >= 0.5 ? wp : wn }
}

/// JSONSerialization crashes on Swift `Float` (it boxes as `__SwiftValue` and raises
/// `NSInvalidArgumentException`, which `try` does not catch). Persist doubles instead.
func jsonDoubles(_ values: [Float]) -> [Double] {
  values.map { $0.isFinite ? Double($0) : 0 }
}

func jsonDoubles2(_ values: [[Float]]) -> [[Double]] {
  values.map(jsonDoubles)
}

func jsonDoubles3(_ values: [[[Float]]]) -> [[[Double]]] {
  values.map(jsonDoubles2)
}

func floatVector(_ any: Any?) -> [Float] {
  guard let arr = any as? [Any] else { return [] }
  return arr.map { ($0 as? NSNumber)?.floatValue ?? 0 }
}

func floatMatrix(_ any: Any?) -> [[Float]] {
  guard let arr = any as? [Any] else { return [] }
  return arr.map { floatVector($0) }
}

func floatTensor(_ any: Any?) -> [[[Float]]] {
  guard let arr = any as? [Any] else { return [] }
  return arr.map { floatMatrix($0) }
}

func sigmoid(_ x: Float) -> Float {
  if x >= 0 {
    let z = exp(-x)
    return 1 / (1 + z)
  }
  let z = exp(x)
  return z / (1 + z)
}

func logLoss(labels: [Float], probs: [Float]) -> Float {
  guard labels.count == probs.count, !labels.isEmpty else { return 0 }
  var sum: Float = 0
  for i in 0..<labels.count {
    let p = min(max(probs[i], 1e-7), 1 - 1e-7)
    let y = labels[i]
    sum -= y * log(p) + (1 - y) * log(1 - p)
  }
  return sum / Float(labels.count)
}

func f1Score(labels: [Float], probs: [Float]) -> Float {
  guard !labels.isEmpty else { return 0 }
  var tp = 0
  var fp = 0
  var fn = 0
  for i in 0..<labels.count {
    let pred = probs[i] >= 0.5
    let positive = labels[i] >= 0.5
    if pred && positive { tp += 1 }
    else if pred && !positive { fp += 1 }
    else if !pred && positive { fn += 1 }
  }
  let precision = tp + fp == 0 ? Float(0) : Float(tp) / Float(tp + fp)
  let recall = tp + fn == 0 ? Float(0) : Float(tp) / Float(tp + fn)
  if precision + recall == 0 { return 0 }
  return 2 * precision * recall / (precision + recall)
}

func accuracy(labels: [Float], probs: [Float]) -> Float {
  guard !labels.isEmpty else { return 0 }
  var ok = 0
  for i in 0..<labels.count {
    let pred = probs[i] >= 0.5 ? Float(1) : Float(0)
    if pred == labels[i] { ok += 1 }
  }
  return Float(ok) / Float(labels.count)
}
