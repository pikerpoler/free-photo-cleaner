import Foundation

// MARK: - Tensor

final class Tensor {
  var data: [Float]
  let shape: [Int]
  var count: Int { data.count }

  init(shape: [Int], fill: Float = 0) {
    self.shape = shape
    let n = shape.reduce(1, *)
    self.data = [Float](repeating: fill, count: n)
  }

  init(shape: [Int], data: [Float]) {
    self.shape = shape
    self.data = data
  }

  func copy() -> Tensor {
    Tensor(shape: shape, data: data)
  }

  subscript(i: Int) -> Float {
    get { data[i] }
    set { data[i] = newValue }
  }
}

// MARK: - Utils

func xavierUniform(fanIn: Int, fanOut: Int, count: Int) -> [Float] {
  let limit = sqrt(6.0 / Float(fanIn + fanOut))
  return (0..<count).map { _ in Float.random(in: -limit...limit) }
}

func heUniform(fanIn: Int, count: Int) -> [Float] {
  let limit = sqrt(6.0 / Float(fanIn))
  return (0..<count).map { _ in Float.random(in: -limit...limit) }
}

func sigmoid(_ x: Float) -> Float {
  if x >= 0 {
    let z = exp(-x)
    return 1 / (1 + z)
  } else {
    let z = exp(x)
    return z / (1 + z)
  }
}

func clampResize(_ v: Int) -> Int {
  min(512, max(32, v))
}

// MARK: - Layers

protocol Layer {
  func forward(_ input: Tensor) -> Tensor
  func backward(_ gradOutput: Tensor) -> Tensor
  func parameters() -> [(weights: UnsafeMutablePointer<Float>, count: Int, bias: UnsafeMutablePointer<Float>?, biasCount: Int)]
  func zeroGrad()
  func applyGrad(lr: Float)
}

final class LinearLayer: Layer {
  var weight: [Float] // [out, in]
  var bias: [Float]
  var weightGrad: [Float]
  var biasGrad: [Float]
  let inFeatures: Int
  let outFeatures: Int
  private var lastInput: Tensor?

  init(inFeatures: Int, outFeatures: Int) {
    self.inFeatures = inFeatures
    self.outFeatures = outFeatures
    self.weight = xavierUniform(fanIn: inFeatures, fanOut: outFeatures, count: inFeatures * outFeatures)
    self.bias = [Float](repeating: 0, count: outFeatures)
    self.weightGrad = [Float](repeating: 0, count: inFeatures * outFeatures)
    self.biasGrad = [Float](repeating: 0, count: outFeatures)
  }

  func forward(_ input: Tensor) -> Tensor {
    lastInput = input
    let out = Tensor(shape: [outFeatures])
    for o in 0..<outFeatures {
      var sum = bias[o]
      let row = o * inFeatures
      for i in 0..<inFeatures {
        sum += weight[row + i] * input[i]
      }
      out[o] = sum
    }
    return out
  }

  func backward(_ gradOutput: Tensor) -> Tensor {
    guard let input = lastInput else { return Tensor(shape: [inFeatures]) }
    let gradInput = Tensor(shape: [inFeatures])
    for o in 0..<outFeatures {
      let g = gradOutput[o]
      biasGrad[o] += g
      let row = o * inFeatures
      for i in 0..<inFeatures {
        weightGrad[row + i] += g * input[i]
        gradInput[i] += g * weight[row + i]
      }
    }
    return gradInput
  }

  func parameters() -> [(weights: UnsafeMutablePointer<Float>, count: Int, bias: UnsafeMutablePointer<Float>?, biasCount: Int)] {
    []
  }

  func zeroGrad() {
    weightGrad = [Float](repeating: 0, count: weightGrad.count)
    biasGrad = [Float](repeating: 0, count: biasGrad.count)
  }

  func applyGrad(lr: Float) {
    for i in 0..<weight.count {
      weight[i] -= lr * weightGrad[i]
    }
    for i in 0..<bias.count {
      bias[i] -= lr * biasGrad[i]
    }
  }
}

final class Conv2dLayer: Layer {
  var weight: [Float] // [outC, inC, kH, kW]
  var bias: [Float]
  var weightGrad: [Float]
  var biasGrad: [Float]
  let inChannels: Int
  let outChannels: Int
  let kernel: Int
  let stride: Int
  let padding: Int
  private var lastInput: Tensor?
  private var outH = 0
  private var outW = 0
  private var inH = 0
  private var inW = 0

  init(inChannels: Int, outChannels: Int, kernel: Int = 3, stride: Int = 1, padding: Int = 1) {
    self.inChannels = inChannels
    self.outChannels = outChannels
    self.kernel = kernel
    self.stride = stride
    self.padding = padding
    let fanIn = inChannels * kernel * kernel
    let wCount = outChannels * fanIn
    self.weight = heUniform(fanIn: fanIn, count: wCount)
    self.bias = [Float](repeating: 0, count: outChannels)
    self.weightGrad = [Float](repeating: 0, count: wCount)
    self.biasGrad = [Float](repeating: 0, count: outChannels)
  }

  private func weightIndex(oc: Int, ic: Int, kh: Int, kw: Int) -> Int {
    ((oc * inChannels + ic) * kernel + kh) * kernel + kw
  }

  func forward(_ input: Tensor) -> Tensor {
    // input shape [C, H, W]
    lastInput = input
    inH = input.shape[1]
    inW = input.shape[2]
    outH = (inH + 2 * padding - kernel) / stride + 1
    outW = (inW + 2 * padding - kernel) / stride + 1
    let out = Tensor(shape: [outChannels, outH, outW])

    for oc in 0..<outChannels {
      for oh in 0..<outH {
        for ow in 0..<outW {
          var sum = bias[oc]
          let ih0 = oh * stride - padding
          let iw0 = ow * stride - padding
          for ic in 0..<inChannels {
            for kh in 0..<kernel {
              let ih = ih0 + kh
              if ih < 0 || ih >= inH { continue }
              for kw in 0..<kernel {
                let iw = iw0 + kw
                if iw < 0 || iw >= inW { continue }
                let inIdx = (ic * inH + ih) * inW + iw
                sum += weight[weightIndex(oc: oc, ic: ic, kh: kh, kw: kw)] * input[inIdx]
              }
            }
          }
          out[(oc * outH + oh) * outW + ow] = sum
        }
      }
    }
    return out
  }

  func backward(_ gradOutput: Tensor) -> Tensor {
    guard let input = lastInput else {
      return Tensor(shape: [inChannels, inH, inW])
    }
    let gradInput = Tensor(shape: [inChannels, inH, inW])

    for oc in 0..<outChannels {
      for oh in 0..<outH {
        for ow in 0..<outW {
          let g = gradOutput[(oc * outH + oh) * outW + ow]
          biasGrad[oc] += g
          let ih0 = oh * stride - padding
          let iw0 = ow * stride - padding
          for ic in 0..<inChannels {
            for kh in 0..<kernel {
              let ih = ih0 + kh
              if ih < 0 || ih >= inH { continue }
              for kw in 0..<kernel {
                let iw = iw0 + kw
                if iw < 0 || iw >= inW { continue }
                let inIdx = (ic * inH + ih) * inW + iw
                let wi = weightIndex(oc: oc, ic: ic, kh: kh, kw: kw)
                weightGrad[wi] += g * input[inIdx]
                gradInput[inIdx] += g * weight[wi]
              }
            }
          }
        }
      }
    }
    return gradInput
  }

  func parameters() -> [(weights: UnsafeMutablePointer<Float>, count: Int, bias: UnsafeMutablePointer<Float>?, biasCount: Int)] { [] }

  func zeroGrad() {
    weightGrad = [Float](repeating: 0, count: weightGrad.count)
    biasGrad = [Float](repeating: 0, count: biasGrad.count)
  }

  func applyGrad(lr: Float) {
    for i in 0..<weight.count { weight[i] -= lr * weightGrad[i] }
    for i in 0..<bias.count { bias[i] -= lr * biasGrad[i] }
  }
}

final class ReLULayer: Layer {
  private var mask: [Bool] = []

  func forward(_ input: Tensor) -> Tensor {
    mask = input.data.map { $0 > 0 }
    let out = Tensor(shape: input.shape)
    for i in 0..<input.count {
      out[i] = mask[i] ? input[i] : 0
    }
    return out
  }

  func backward(_ gradOutput: Tensor) -> Tensor {
    let out = Tensor(shape: gradOutput.shape)
    for i in 0..<gradOutput.count {
      out[i] = mask[i] ? gradOutput[i] : 0
    }
    return out
  }

  func parameters() -> [(weights: UnsafeMutablePointer<Float>, count: Int, bias: UnsafeMutablePointer<Float>?, biasCount: Int)] { [] }
  func zeroGrad() {}
  func applyGrad(lr: Float) {}
}

final class MaxPool2dLayer: Layer {
  let kernel: Int
  let stride: Int
  private var switches: [(c: Int, h: Int, w: Int)] = []
  private var inShape: [Int] = []
  private var outH = 0
  private var outW = 0

  init(kernel: Int = 2, stride: Int = 2) {
    self.kernel = kernel
    self.stride = stride
  }

  func forward(_ input: Tensor) -> Tensor {
    // [C,H,W]
    inShape = input.shape
    let c = input.shape[0]
    let h = input.shape[1]
    let w = input.shape[2]
    outH = h / stride
    outW = w / stride
    let out = Tensor(shape: [c, outH, outW])
    switches = Array(repeating: (0, 0, 0), count: c * outH * outW)

    for ci in 0..<c {
      for oh in 0..<outH {
        for ow in 0..<outW {
          let h0 = oh * stride
          let w0 = ow * stride
          var maxV = -Float.greatestFiniteMagnitude
          var maxH = h0
          var maxW = w0
          for kh in 0..<kernel {
            for kw in 0..<kernel {
              let ih = h0 + kh
              let iw = w0 + kw
              if ih >= h || iw >= w { continue }
              let v = input[(ci * h + ih) * w + iw]
              if v > maxV {
                maxV = v
                maxH = ih
                maxW = iw
              }
            }
          }
          let oi = (ci * outH + oh) * outW + ow
          out[oi] = maxV
          switches[oi] = (ci, maxH, maxW)
        }
      }
    }
    return out
  }

  func backward(_ gradOutput: Tensor) -> Tensor {
    let c = inShape[0]
    let h = inShape[1]
    let w = inShape[2]
    let gradInput = Tensor(shape: inShape)
    for i in 0..<switches.count {
      let sw = switches[i]
      let idx = (sw.c * h + sw.h) * w + sw.w
      gradInput[idx] += gradOutput[i]
    }
    return gradInput
  }

  func parameters() -> [(weights: UnsafeMutablePointer<Float>, count: Int, bias: UnsafeMutablePointer<Float>?, biasCount: Int)] { [] }
  func zeroGrad() {}
  func applyGrad(lr: Float) {}
}

final class GlobalAvgPool2dLayer: Layer {
  private var inShape: [Int] = []

  func forward(_ input: Tensor) -> Tensor {
    // [C,H,W] -> [C]
    inShape = input.shape
    let c = input.shape[0]
    let h = input.shape[1]
    let w = input.shape[2]
    let spatial = Float(h * w)
    let out = Tensor(shape: [c])
    for ci in 0..<c {
      var sum: Float = 0
      let base = ci * h * w
      for i in 0..<(h * w) {
        sum += input[base + i]
      }
      out[ci] = sum / spatial
    }
    return out
  }

  func backward(_ gradOutput: Tensor) -> Tensor {
    let c = inShape[0]
    let h = inShape[1]
    let w = inShape[2]
    let spatial = Float(h * w)
    let gradInput = Tensor(shape: inShape)
    for ci in 0..<c {
      let g = gradOutput[ci] / spatial
      let base = ci * h * w
      for i in 0..<(h * w) {
        gradInput[base + i] = g
      }
    }
    return gradInput
  }

  func parameters() -> [(weights: UnsafeMutablePointer<Float>, count: Int, bias: UnsafeMutablePointer<Float>?, biasCount: Int)] { [] }
  func zeroGrad() {}
  func applyGrad(lr: Float) {}
}

final class FlattenLayer: Layer {
  private var inShape: [Int] = []

  func forward(_ input: Tensor) -> Tensor {
    inShape = input.shape
    return Tensor(shape: [input.count], data: input.data)
  }

  func backward(_ gradOutput: Tensor) -> Tensor {
    return Tensor(shape: inShape, data: gradOutput.data)
  }

  func parameters() -> [(weights: UnsafeMutablePointer<Float>, count: Int, bias: UnsafeMutablePointer<Float>?, biasCount: Int)] { [] }
  func zeroGrad() {}
  func applyGrad(lr: Float) {}
}

// MARK: - Residual block

final class BasicBlock: Layer {
  let conv1: Conv2dLayer
  let relu1: ReLULayer
  let conv2: Conv2dLayer
  let downsample: Conv2dLayer?
  let reluOut: ReLULayer

  init(inChannels: Int, outChannels: Int, stride: Int) {
    conv1 = Conv2dLayer(inChannels: inChannels, outChannels: outChannels, kernel: 3, stride: stride, padding: 1)
    relu1 = ReLULayer()
    conv2 = Conv2dLayer(inChannels: outChannels, outChannels: outChannels, kernel: 3, stride: 1, padding: 1)
    if stride != 1 || inChannels != outChannels {
      downsample = Conv2dLayer(inChannels: inChannels, outChannels: outChannels, kernel: 1, stride: stride, padding: 0)
    } else {
      downsample = nil
    }
    reluOut = ReLULayer()
  }

  func forward(_ input: Tensor) -> Tensor {
    var out = conv1.forward(input)
    out = relu1.forward(out)
    out = conv2.forward(out)
    let residual: Tensor
    if let ds = downsample {
      residual = ds.forward(input)
    } else {
      residual = input
    }
    let summed = Tensor(shape: out.shape)
    for i in 0..<out.count {
      summed[i] = out[i] + residual[i]
    }
    return reluOut.forward(summed)
  }

  func backward(_ gradOutput: Tensor) -> Tensor {
    var g = reluOut.backward(gradOutput)
    let gMain = conv2.backward(g)
    var gRelu = relu1.backward(gMain)
    let gConv1 = conv1.backward(gRelu)
    if let ds = downsample {
      let gDs = ds.backward(g)
      let combined = Tensor(shape: gConv1.shape)
      for i in 0..<gConv1.count {
        combined[i] = gConv1[i] + gDs[i]
      }
      return combined
    }
    // identity residual
    for i in 0..<gConv1.count {
      gConv1[i] += g[i]
    }
    return gConv1
  }

  func parameters() -> [(weights: UnsafeMutablePointer<Float>, count: Int, bias: UnsafeMutablePointer<Float>?, biasCount: Int)] { [] }

  func zeroGrad() {
    conv1.zeroGrad()
    conv2.zeroGrad()
    downsample?.zeroGrad()
  }

  func applyGrad(lr: Float) {
    conv1.applyGrad(lr: lr)
    conv2.applyGrad(lr: lr)
    downsample?.applyGrad(lr: lr)
  }

  var allParamLayers: [Layer] {
    var layers: [Layer] = [conv1, conv2]
    if let ds = downsample { layers.append(ds) }
    return layers
  }
}

// MARK: - Model protocol

protocol TrainableModel: AnyObject {
  func forward(_ input: Tensor) -> Float
  func backward(target: Float)
  func zeroGrad()
  func applyGrad(lr: Float, batchSize: Int)
  func serialize() -> Data
  static func deserialize(_ data: Data, inputSize: Int) -> Self?
  var modelId: String { get }
}

private func bceLoss(pred: Float, target: Float) -> Float {
  let p = min(max(pred, 1e-7), 1 - 1e-7)
  return -(target * log(p) + (1 - target) * log(1 - p))
}

private func bceGrad(pred: Float, target: Float) -> Float {
  // dL/dz where pred = sigmoid(z), for BCE: pred - target
  return pred - target
}

// MARK: - Sequential CNN / MLP

final class SequentialModel: TrainableModel {
  let modelId: String
  let layers: [Layer]
  private var lastPred: Float = 0.5
  private var lastLogit: Float = 0

  init(modelId: String, layers: [Layer]) {
    self.modelId = modelId
    self.layers = layers
  }

  func forward(_ input: Tensor) -> Float {
    var x = input
    for layer in layers {
      x = layer.forward(x)
    }
    lastLogit = x[0]
    lastPred = sigmoid(lastLogit)
    return lastPred
  }

  func backward(target: Float) {
    let dLossDz = bceGrad(pred: lastPred, target: target)
    var g = Tensor(shape: [1], data: [dLossDz])
    for layer in layers.reversed() {
      g = layer.backward(g)
    }
  }

  func zeroGrad() {
    for layer in layers { layer.zeroGrad() }
  }

  func applyGrad(lr: Float, batchSize: Int) {
    let scale = lr / Float(max(1, batchSize))
    for layer in layers { layer.applyGrad(lr: scale) }
  }

  func serialize() -> Data {
    var parts: [Data] = []
    let idData = Data(modelId.utf8)
    var idLen = UInt32(idData.count)
    parts.append(Data(bytes: &idLen, count: 4))
    parts.append(idData)

    func appendFloats(_ arr: [Float]) {
      arr.withUnsafeBufferPointer { buf in
        parts.append(Data(buffer: buf))
      }
    }

    for layer in layers {
      if let linear = layer as? LinearLayer {
        var tag: UInt32 = 1
        var o = UInt32(linear.outFeatures)
        var i = UInt32(linear.inFeatures)
        parts.append(Data(bytes: &tag, count: 4))
        parts.append(Data(bytes: &o, count: 4))
        parts.append(Data(bytes: &i, count: 4))
        appendFloats(linear.weight)
        appendFloats(linear.bias)
      } else if let conv = layer as? Conv2dLayer {
        var tag: UInt32 = 2
        var oc = UInt32(conv.outChannels)
        var ic = UInt32(conv.inChannels)
        var k = UInt32(conv.kernel)
        var s = UInt32(conv.stride)
        var p = UInt32(conv.padding)
        parts.append(Data(bytes: &tag, count: 4))
        parts.append(Data(bytes: &oc, count: 4))
        parts.append(Data(bytes: &ic, count: 4))
        parts.append(Data(bytes: &k, count: 4))
        parts.append(Data(bytes: &s, count: 4))
        parts.append(Data(bytes: &p, count: 4))
        appendFloats(conv.weight)
        appendFloats(conv.bias)
      } else if layer is ReLULayer {
        var tag: UInt32 = 3
        parts.append(Data(bytes: &tag, count: 4))
      } else if let pool = layer as? MaxPool2dLayer {
        var tag: UInt32 = 4
        var k = UInt32(pool.kernel)
        var s = UInt32(pool.stride)
        parts.append(Data(bytes: &tag, count: 4))
        parts.append(Data(bytes: &k, count: 4))
        parts.append(Data(bytes: &s, count: 4))
      } else if layer is FlattenLayer {
        var tag: UInt32 = 5
        parts.append(Data(bytes: &tag, count: 4))
      } else if layer is GlobalAvgPool2dLayer {
        var tag: UInt32 = 6
        parts.append(Data(bytes: &tag, count: 4))
      }
    }
    var end: UInt32 = 0
    parts.append(Data(bytes: &end, count: 4))
    return parts.reduce(Data(), +)
  }

  static func deserialize(_ data: Data, inputSize: Int) -> SequentialModel? {
    // Models are rebuilt from modelId + weight blobs via factory; simple path stores full rebuild
    return nil
  }
}

// MARK: - ResNet-18

final class ResNet18Model: TrainableModel {
  let modelId = "resnet-18"
  let stem: Conv2dLayer
  let stemRelu: ReLULayer
  let layers: [BasicBlock]
  let pool: GlobalAvgPool2dLayer
  let fc: LinearLayer
  private var lastPred: Float = 0.5

  init(inputChannels: Int = 3) {
    // Small-image friendly stem (3x3 s1) instead of 7x7 s2
    stem = Conv2dLayer(inChannels: inputChannels, outChannels: 64, kernel: 3, stride: 1, padding: 1)
    stemRelu = ReLULayer()
    var blocks: [BasicBlock] = []
    // layer1: 64
    blocks.append(BasicBlock(inChannels: 64, outChannels: 64, stride: 1))
    blocks.append(BasicBlock(inChannels: 64, outChannels: 64, stride: 1))
    // layer2: 128
    blocks.append(BasicBlock(inChannels: 64, outChannels: 128, stride: 2))
    blocks.append(BasicBlock(inChannels: 128, outChannels: 128, stride: 1))
    // layer3: 256
    blocks.append(BasicBlock(inChannels: 128, outChannels: 256, stride: 2))
    blocks.append(BasicBlock(inChannels: 256, outChannels: 256, stride: 1))
    // layer4: 512
    blocks.append(BasicBlock(inChannels: 256, outChannels: 512, stride: 2))
    blocks.append(BasicBlock(inChannels: 512, outChannels: 512, stride: 1))
    layers = blocks
    pool = GlobalAvgPool2dLayer()
    fc = LinearLayer(inFeatures: 512, outFeatures: 1)
  }

  func forward(_ input: Tensor) -> Float {
    var x = stem.forward(input)
    x = stemRelu.forward(x)
    for block in layers {
      x = block.forward(x)
    }
    x = pool.forward(x)
    x = fc.forward(x)
    lastPred = sigmoid(x[0])
    return lastPred
  }

  func backward(target: Float) {
    let dLossDz = bceGrad(pred: lastPred, target: target)
    var g = Tensor(shape: [1], data: [dLossDz])
    g = fc.backward(g)
    g = pool.backward(g)
    for block in layers.reversed() {
      g = block.backward(g)
    }
    g = stemRelu.backward(g)
    _ = stem.backward(g)
  }

  func zeroGrad() {
    stem.zeroGrad()
    for block in layers { block.zeroGrad() }
    fc.zeroGrad()
  }

  func applyGrad(lr: Float, batchSize: Int) {
    let scale = lr / Float(max(1, batchSize))
    stem.applyGrad(lr: scale)
    for block in layers { block.applyGrad(lr: scale) }
    fc.applyGrad(lr: scale)
  }

  func serialize() -> Data {
    var parts: [Data] = []
    func appendFloats(_ arr: [Float]) {
      arr.withUnsafeBufferPointer { buf in
        parts.append(Data(buffer: buf))
      }
    }
    func appendConv(_ conv: Conv2dLayer) {
      var oc = UInt32(conv.outChannels)
      var ic = UInt32(conv.inChannels)
      var k = UInt32(conv.kernel)
      var s = UInt32(conv.stride)
      var p = UInt32(conv.padding)
      parts.append(Data(bytes: &oc, count: 4))
      parts.append(Data(bytes: &ic, count: 4))
      parts.append(Data(bytes: &k, count: 4))
      parts.append(Data(bytes: &s, count: 4))
      parts.append(Data(bytes: &p, count: 4))
      appendFloats(conv.weight)
      appendFloats(conv.bias)
    }
    func appendLinear(_ lin: LinearLayer) {
      var o = UInt32(lin.outFeatures)
      var i = UInt32(lin.inFeatures)
      parts.append(Data(bytes: &o, count: 4))
      parts.append(Data(bytes: &i, count: 4))
      appendFloats(lin.weight)
      appendFloats(lin.bias)
    }
    let magic = Data("RS18".utf8)
    parts.append(magic)
    appendConv(stem)
    for block in layers {
      appendConv(block.conv1)
      appendConv(block.conv2)
      if let ds = block.downsample {
        var has: UInt32 = 1
        parts.append(Data(bytes: &has, count: 4))
        appendConv(ds)
      } else {
        var has: UInt32 = 0
        parts.append(Data(bytes: &has, count: 4))
      }
    }
    appendLinear(fc)
    return parts.reduce(Data(), +)
  }

  static func deserialize(_ data: Data, inputSize: Int) -> ResNet18Model? {
    guard data.count > 4, String(data: data.prefix(4), encoding: .utf8) == "RS18" else { return nil }
    let model = ResNet18Model()
    var offset = 4

    func readU32() -> UInt32 {
      let v: UInt32 = data.subdata(in: offset..<(offset + 4)).withUnsafeBytes { $0.load(as: UInt32.self) }
      offset += 4
      return v
    }
    func readFloats(_ n: Int) -> [Float] {
      let byteCount = n * 4
      let slice = data.subdata(in: offset..<(offset + byteCount))
      offset += byteCount
      return slice.withUnsafeBytes { buf in
        Array(buf.bindMemory(to: Float.self))
      }
    }
    func loadConv(_ conv: Conv2dLayer) {
      _ = readU32(); _ = readU32(); _ = readU32(); _ = readU32(); _ = readU32()
      conv.weight = readFloats(conv.weight.count)
      conv.bias = readFloats(conv.bias.count)
    }
    func loadLinear(_ lin: LinearLayer) {
      _ = readU32(); _ = readU32()
      lin.weight = readFloats(lin.weight.count)
      lin.bias = readFloats(lin.bias.count)
    }

    loadConv(model.stem)
    for block in model.layers {
      loadConv(block.conv1)
      loadConv(block.conv2)
      let has = readU32()
      if has == 1, let ds = block.downsample {
        loadConv(ds)
      }
    }
    loadLinear(model.fc)
    return model
  }
}

// MARK: - Factory

enum ModelFactory {
  static func create(modelId: String, resize: Int) -> TrainableModel {
    let r = clampResize(resize)
    let flat = 3 * r * r

    switch modelId {
    case "mlp-tiny":
      return SequentialModel(modelId: modelId, layers: [
        FlattenLayer(),
        LinearLayer(inFeatures: flat, outFeatures: 1),
      ])
    case "mlp-small":
      return SequentialModel(modelId: modelId, layers: [
        FlattenLayer(),
        LinearLayer(inFeatures: flat, outFeatures: 64),
        ReLULayer(),
        LinearLayer(inFeatures: 64, outFeatures: 1),
      ])
    case "mlp-medium":
      return SequentialModel(modelId: modelId, layers: [
        FlattenLayer(),
        LinearLayer(inFeatures: flat, outFeatures: 256),
        ReLULayer(),
        LinearLayer(inFeatures: 256, outFeatures: 64),
        ReLULayer(),
        LinearLayer(inFeatures: 64, outFeatures: 1),
      ])
    case "cnn-nano":
      return SequentialModel(modelId: modelId, layers: [
        Conv2dLayer(inChannels: 3, outChannels: 8, kernel: 3, stride: 1, padding: 1),
        ReLULayer(),
        MaxPool2dLayer(kernel: 2, stride: 2),
        Conv2dLayer(inChannels: 8, outChannels: 16, kernel: 3, stride: 1, padding: 1),
        ReLULayer(),
        MaxPool2dLayer(kernel: 2, stride: 2),
        GlobalAvgPool2dLayer(),
        LinearLayer(inFeatures: 16, outFeatures: 1),
      ])
    case "cnn-tiny":
      return SequentialModel(modelId: modelId, layers: [
        Conv2dLayer(inChannels: 3, outChannels: 16, kernel: 3, stride: 1, padding: 1),
        ReLULayer(),
        MaxPool2dLayer(kernel: 2, stride: 2),
        Conv2dLayer(inChannels: 16, outChannels: 32, kernel: 3, stride: 1, padding: 1),
        ReLULayer(),
        MaxPool2dLayer(kernel: 2, stride: 2),
        Conv2dLayer(inChannels: 32, outChannels: 64, kernel: 3, stride: 1, padding: 1),
        ReLULayer(),
        MaxPool2dLayer(kernel: 2, stride: 2),
        GlobalAvgPool2dLayer(),
        LinearLayer(inFeatures: 64, outFeatures: 1),
      ])
    case "cnn-small":
      return SequentialModel(modelId: modelId, layers: [
        Conv2dLayer(inChannels: 3, outChannels: 32, kernel: 3, stride: 1, padding: 1),
        ReLULayer(),
        MaxPool2dLayer(kernel: 2, stride: 2),
        Conv2dLayer(inChannels: 32, outChannels: 64, kernel: 3, stride: 1, padding: 1),
        ReLULayer(),
        MaxPool2dLayer(kernel: 2, stride: 2),
        Conv2dLayer(inChannels: 64, outChannels: 128, kernel: 3, stride: 1, padding: 1),
        ReLULayer(),
        MaxPool2dLayer(kernel: 2, stride: 2),
        Conv2dLayer(inChannels: 128, outChannels: 128, kernel: 3, stride: 1, padding: 1),
        ReLULayer(),
        GlobalAvgPool2dLayer(),
        LinearLayer(inFeatures: 128, outFeatures: 1),
      ])
    case "resnet-18":
      return ResNet18Model()
    default:
      return SequentialModel(modelId: "cnn-nano", layers: [
        Conv2dLayer(inChannels: 3, outChannels: 8, kernel: 3, stride: 1, padding: 1),
        ReLULayer(),
        MaxPool2dLayer(kernel: 2, stride: 2),
        GlobalAvgPool2dLayer(),
        LinearLayer(inFeatures: 8, outFeatures: 1),
      ])
    }
  }
}

func computeBCELoss(pred: Float, target: Float) -> Float {
  bceLoss(pred: pred, target: target)
}
