import Foundation
import Accelerate

/// The 16 photometric features computed from an RGBA8888 buffer.
/// Kept free of UIKit so it can be compiled and tested on macOS.
enum PhotometricCore {
  static func compute(
    rgba: UnsafePointer<UInt8>,
    width w: Int,
    height h: Int,
    bytesPerRow: Int
  ) -> [Float] {
    let count = w * h
    guard w > 0, h > 0 else {
      return [Float](repeating: 0, count: 16)
    }

    // Pass 1: per-pixel stats.
    var lum = [Float](repeating: 0, count: count)
    var hist = [Int](repeating: 0, count: 32)
    var sumL = 0.0, sumL2 = 0.0
    var sumS = 0.0, sumS2 = 0.0
    var sumRG = 0.0, sumRG2 = 0.0
    var sumYB = 0.0, sumYB2 = 0.0
    var sumR = 0.0, sumB = 0.0
    var darkCount = 0, brightCount = 0
    lum.withUnsafeMutableBufferPointer { lp in
      hist.withUnsafeMutableBufferPointer { hp in
        for y in 0..<h {
          let row = rgba + y * bytesPerRow
          let base = y * w
          for x in 0..<w {
            let p = row + x * 4
            let r = Float(p[0]) / 255
            let g = Float(p[1]) / 255
            let b = Float(p[2]) / 255
            let l = 0.2126 * r + 0.7152 * g + 0.0722 * b
            lp[base + x] = l
            let mx = max(r, max(g, b))
            let mn = min(r, min(g, b))
            let s = mx > 1e-6 ? (mx - mn) / mx : 0
            let rg = r - g
            let yb = 0.5 * (r + g) - b
            let ld = Double(l), sd = Double(s), rgd = Double(rg), ybd = Double(yb)
            sumL += ld; sumL2 += ld * ld
            sumS += sd; sumS2 += sd * sd
            sumRG += rgd; sumRG2 += rgd * rgd
            sumYB += ybd; sumYB2 += ybd * ybd
            sumR += Double(r); sumB += Double(b)
            if l < 0.05 { darkCount += 1 }
            if l > 0.95 { brightCount += 1 }
            hp[min(31, max(0, Int(l * 32)))] += 1
          }
        }
      }
    }
    let nd = Double(count)
    let n = Float(count)
    func meanStd(_ sum: Double, _ sumSq: Double) -> (Float, Float) {
      let m = sum / nd
      return (Float(m), Float(max(0, sumSq / nd - m * m).squareRoot()))
    }
    let (meanL, stdL) = meanStd(sumL, sumL2)
    let (meanSat, stdSat) = meanStd(sumS, sumS2)
    let (rgMean, rgStd) = meanStd(sumRG, sumRG2)
    let (ybMean, ybStd) = meanStd(sumYB, sumYB2)
    let colorfulness = sqrt(rgStd * rgStd + ybStd * ybStd) + 0.3 * sqrt(rgMean * rgMean + ybMean * ybMean)
    let warmCool = Float(sumR / nd) - Float(sumB / nd)

    var entropy: Float = 0
    for c in hist where c > 0 {
      let p = Float(c) / n
      entropy -= p * log2(p)
    }
    // Thresholds 0.25/0.5/0.75 are exact histogram bin edges (bins of width 1/32).
    func binRange(_ lo: Int, _ hi: Int) -> Float {
      Float(hist[lo..<hi].reduce(0, +)) / n
    }
    let q1 = binRange(0, 8)
    let q2 = binRange(8, 16)
    let q3 = binRange(16, 24)
    let q4 = binRange(24, 32)

    // Pass 2: Laplacian (overall + center/border) and 3x3 high-pass residual.
    var lapVar: Float = 0
    var ratio: Float = 0
    var noiseMad: Float = 0
    if w > 2, h > 2 {
      let y0 = h / 4, y1 = h - h / 4
      let x0 = w / 4, x1 = w - w / 4
      var cSum = 0.0, cSum2 = 0.0, cN = 0
      var bSum = 0.0, bSum2 = 0.0, bN = 0
      let highCount = (w - 2) * (h - 2)
      var high = [Float](repeating: 0, count: highCount)
      lum.withUnsafeBufferPointer { lp in
        high.withUnsafeMutableBufferPointer { hp in
          var k = 0
          for y in 1..<(h - 1) {
            let inRows = y >= y0 && y < y1
            for x in 1..<(w - 1) {
              let i = y * w + x
              let c = lp[i]
              let up = lp[i - w]
              let down = lp[i + w]
              let left = lp[i - 1]
              let right = lp[i + 1]
              let lap = -4 * c + up + down + left + right
              let lapD = Double(lap)
              if inRows && x >= x0 && x < x1 {
                cSum += lapD; cSum2 += lapD * lapD; cN += 1
              } else {
                bSum += lapD; bSum2 += lapD * lapD; bN += 1
              }
              var sum: Float = 0
              sum += lp[i - w - 1]
              sum += up
              sum += lp[i - w + 1]
              sum += left
              sum += c
              sum += right
              sum += lp[i + w - 1]
              sum += down
              sum += lp[i + w + 1]
              hp[k] = c - sum / 9
              k += 1
            }
          }
        }
      }
      func variance(_ sum: Double, _ sumSq: Double, _ cnt: Int) -> Double {
        guard cnt > 0 else { return 0 }
        let m = sum / Double(cnt)
        return max(0, sumSq / Double(cnt) - m * m)
      }
      lapVar = Float(variance(cSum + bSum, cSum2 + bSum2, cN + bN))
      let centerVar = Float(variance(cSum, cSum2, cN))
      let borderVar = Float(variance(bSum, bSum2, bN))
      ratio = centerVar / (borderVar + 1e-8)

      noiseMad = high.withUnsafeMutableBufferPointer { hp -> Float in
        let p = hp.baseAddress!
        let len = vDSP_Length(highCount)
        vDSP_vsort(p, len, 1)
        var negMed = -p[highCount / 2]
        vDSP_vsadd(p, 1, &negMed, p, 1, len)
        vDSP_vabs(p, 1, p, 1, len)
        vDSP_vsort(p, len, 1)
        return p[highCount / 2]
      }
    }

    return [
      meanL, stdL, Float(darkCount) / n, Float(brightCount) / n,
      lapVar, ratio, meanSat, stdSat, colorfulness, warmCool,
      entropy, noiseMad, q1, q2, q3, q4,
    ]
  }
}