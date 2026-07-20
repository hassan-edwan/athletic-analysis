// Keypoint trajectory conditioning (port of filtering.py): gap interpolation,
// spike removal, zero-lag Butterworth smoothing, confidence median filter.
//
// The Butterworth design + filtfilt replicate scipy.signal.butter/filtfilt
// (digital low-pass via bilinear transform, odd-reflection padding, lfilter_zi
// initial conditions) so smoothed trajectories match the Python core within
// floating-point tolerance.

import Foundation

// Minimal complex arithmetic for the filter design (no external deps).
private struct Complex {
    var re: Double
    var im: Double

    static func + (l: Complex, r: Complex) -> Complex { .init(re: l.re + r.re, im: l.im + r.im) }
    static func - (l: Complex, r: Complex) -> Complex { .init(re: l.re - r.re, im: l.im - r.im) }
    static func * (l: Complex, r: Complex) -> Complex {
        .init(re: l.re * r.re - l.im * r.im, im: l.re * r.im + l.im * r.re)
    }
    static func / (l: Complex, r: Complex) -> Complex {
        let d = r.re * r.re + r.im * r.im
        return .init(re: (l.re * r.re + l.im * r.im) / d,
                     im: (l.im * r.re - l.re * r.im) / d)
    }
    static func scale(_ s: Double, _ c: Complex) -> Complex { .init(re: s * c.re, im: s * c.im) }
}

/// Expand a monic polynomial from its roots; returns real coefficients
/// (roots come in conjugate pairs, so imaginary parts cancel).
private func polyFromRoots(_ roots: [Complex]) -> [Double] {
    var coeffs: [Complex] = [Complex(re: 1, im: 0)]
    for r in roots {
        var next = [Complex](repeating: Complex(re: 0, im: 0), count: coeffs.count + 1)
        for (i, c) in coeffs.enumerated() {
            next[i] = next[i] + c
            next[i + 1] = next[i + 1] - c * r
        }
        coeffs = next
    }
    return coeffs.map(\.re)
}

enum Butterworth {
    /// scipy.signal.butter(order, wn) digital low-pass; wn in (0, 1) as a
    /// fraction of Nyquist. Returns (b, a), each order+1 long.
    static func design(order: Int, wn: Double) -> (b: [Double], a: [Double]) {
        let fs = 2.0
        let warped = 2.0 * fs * tan(Double.pi * wn / fs)
        // Analog Butterworth prototype poles (buttap): for k = 1…order the
        // angle pi*(2k+order-1)/(2*order) already lands in the left half-plane,
        // matching scipy's -exp(j*pi*m/(2N)) with m = -N+1, -N+3, … N-1.
        var poles: [Complex] = []
        for k in 1...order {
            let theta = Double.pi * Double(2 * k + order - 1) / Double(2 * order)
            poles.append(Complex(re: cos(theta), im: sin(theta)))  // exp(j*theta)
        }
        // lp2lp: scale poles by cutoff; gain k = warped^order (no zeros).
        poles = poles.map { Complex.scale(warped, $0) }
        var gain = pow(warped, Double(order))
        // Bilinear transform (bilinear_zpk with fs = 2).
        let fs2 = Complex(re: 2.0 * fs, im: 0)
        var num = Complex(re: 1, im: 0)  // prod(fs2 - z) with no zeros
        var den = Complex(re: 1, im: 0)
        var zPoles: [Complex] = []
        for p in poles {
            zPoles.append((fs2 + p) / (fs2 - p))
            den = den * (fs2 - p)
        }
        gain *= (num / den).re
        num = Complex(re: 1, im: 0)  // silence "written but never read" note
        _ = num
        // Zeros at z = -1 with multiplicity `order`.
        let zZeros = [Complex](repeating: Complex(re: -1, im: 0), count: order)
        let b = polyFromRoots(zZeros).map { $0 * gain }
        let a = polyFromRoots(zPoles)
        return (b, a)
    }
}

enum Filtfilt {
    /// scipy.signal.lfilter (direct form II transposed) with initial state zi.
    static func lfilter(b: [Double], a: [Double], x: [Double],
                        zi: [Double]) -> (y: [Double], zf: [Double]) {
        let n = b.count  // == a.count, a[0] == 1
        var z = zi
        var y = [Double](repeating: 0, count: x.count)
        for i in 0..<x.count {
            let xi = x[i]
            let yi = b[0] * xi + z[0]
            for j in 0..<(n - 2) {
                z[j] = b[j + 1] * xi + z[j + 1] - a[j + 1] * yi
            }
            z[n - 2] = b[n - 1] * xi - a[n - 1] * yi
            y[i] = yi
        }
        return (y, z)
    }

    /// scipy.signal.lfilter_zi: steady-state initial conditions.
    static func lfilterZi(b: [Double], a: [Double]) -> [Double] {
        let n = b.count  // filters here always have len(a) == len(b)
        let m = n - 1
        // IminusA = eye(m) - companion(a).T
        // companion(a): first row -a[1:]/a[0], sub-diagonal ones.
        var mat = [[Double]](repeating: [Double](repeating: 0, count: m), count: m)
        for i in 0..<m {
            for j in 0..<m {
                var companionT = 0.0
                if j == 0 { companionT = -a[i + 1] / a[0] }       // transposed first row
                else if i == j - 1 { companionT = 1.0 }           // transposed sub-diagonal
                mat[i][j] = (i == j ? 1.0 : 0.0) - companionT
            }
        }
        var rhs = (0..<m).map { b[$0 + 1] - a[$0 + 1] * b[0] }
        // Gaussian elimination with partial pivoting.
        for col in 0..<m {
            var pivot = col
            for row in (col + 1)..<m where abs(mat[row][col]) > abs(mat[pivot][col]) {
                pivot = row
            }
            mat.swapAt(col, pivot)
            rhs.swapAt(col, pivot)
            let diag = mat[col][col]
            for row in (col + 1)..<m {
                let factor = mat[row][col] / diag
                if factor == 0 { continue }
                for k in col..<m { mat[row][k] -= factor * mat[col][k] }
                rhs[row] -= factor * rhs[col]
            }
        }
        var zi = [Double](repeating: 0, count: m)
        for row in stride(from: m - 1, through: 0, by: -1) {
            var sum = rhs[row]
            for k in (row + 1)..<m { sum -= mat[row][k] * zi[k] }
            zi[row] = sum / mat[row][row]
        }
        return zi
    }

    /// scipy.signal.filtfilt with the default odd-reflection padding and
    /// scipy's default padlen = 3 * max(len(a), len(b)).
    static func apply(b: [Double], a: [Double], x: [Double]) -> [Double] {
        let padlen = 3 * max(a.count, b.count)
        precondition(x.count > padlen, "signal too short for filtfilt")
        var ext = [Double]()
        ext.reserveCapacity(x.count + 2 * padlen)
        for i in stride(from: padlen, through: 1, by: -1) {
            ext.append(2 * x[0] - x[i])
        }
        ext.append(contentsOf: x)
        let n = x.count
        for i in 2...(padlen + 1) {
            ext.append(2 * x[n - 1] - x[n - i])
        }
        let zi = lfilterZi(b: b, a: a)
        var (y, _) = lfilter(b: b, a: a, x: ext, zi: zi.map { $0 * ext[0] })
        y.reverse()
        (y, _) = lfilter(b: b, a: a, x: y, zi: zi.map { $0 * y[0] })
        y.reverse()
        return Array(y[padlen..<(y.count - padlen)])
    }
}

public enum Filtering {
    public static let minConf = 0.3

    /// Linearly interpolate over invalid samples; hold ends (fill_gaps).
    static func fillGaps(_ values: [Double], valid: [Bool]) -> [Double] {
        if valid.allSatisfy({ $0 }) || !valid.contains(true) { return values }
        var xp = [Double]()
        var fp = [Double]()
        for (i, ok) in valid.enumerated() where ok {
            xp.append(Double(i))
            fp.append(values[i])
        }
        var out = values
        var xq = [Double]()
        var targets = [Int]()
        for (i, ok) in valid.enumerated() where !ok {
            xq.append(Double(i))
            targets.append(i)
        }
        let filled = NaNMath.interp(xq, xp: xp, fp: fp)
        for (t, v) in zip(targets, filled) { out[t] = v }
        return out
    }

    /// Drop single-frame position spikes and re-interpolate (remove_spikes).
    static func removeSpikes(x: [Double], y: [Double],
                             factor: Double = 8.0) -> (x: [Double], y: [Double]) {
        let n = x.count
        guard n >= 3 else { return (x, y) }
        var d = [Double]()
        d.reserveCapacity(n - 1)
        for i in 0..<(n - 1) {
            d.append((pow(x[i + 1] - x[i], 2) + pow(y[i + 1] - y[i], 2)).squareRoot())
        }
        let moving = d.filter { $0 > 1e-9 }
        guard moving.count >= 4 else { return (x, y) }
        let thresh = factor * NaNMath.nanMedian(moving)
        guard thresh > 1e-9 else { return (x, y) }
        var bad = [Bool](repeating: false, count: n)
        for i in 1..<(n - 1) where d[i - 1] > thresh && d[i] > thresh {
            bad[i] = true
        }
        guard bad.contains(true), bad.contains(false) else { return (x, y) }
        let good = bad.map { !$0 }
        return (fillGaps(x, valid: good), fillGaps(y, valid: good))
    }

    /// Zero-lag Butterworth low-pass (lowpass); falls back to a small moving
    /// average when the signal is too short for filtfilt.
    static func lowpass(_ signal: [Double], fps: Double,
                        cutoffHz: Double = 6.0, order: Int = 4) -> [Double] {
        let nyq = fps / 2.0
        let cutoff = min(cutoffHz, nyq * 0.95)
        let (b, a) = Butterworth.design(order: order, wn: cutoff / nyq)
        let padlen = 3 * (max(a.count, b.count) - 1)
        if signal.count <= padlen {
            // np.convolve(signal, ones(k)/k, mode="same")
            let k = max(1, min(5, signal.count))
            let n = signal.count
            var out = [Double](repeating: 0, count: n)
            let offset = (k - 1) / 2  // 'same' alignment for odd/even kernels
            for i in 0..<n {
                var sum = 0.0
                for j in 0..<k {
                    let idx = i + offset - j
                    if idx >= 0 && idx < n { sum += signal[idx] }
                }
                out[i] = sum / Double(k)
            }
            return out
        }
        return Filtfilt.apply(b: b, a: a, x: signal)
    }

    /// Condition raw per-frame keypoints (smooth_keypoints): x/y interpolated
    /// across low-confidence gaps then low-pass filtered; confidence
    /// median-filtered so overlay bones don't flicker.
    public static func smoothKeypoints(_ kpts: PoseSequence, fps: Double,
                                       cutoffHz: Double = 6.0,
                                       minConf: Double = Filtering.minConf) -> PoseSequence {
        let T = kpts.count
        guard T >= 3 else { return kpts }
        var out = kpts
        for k in KP.allCases {
            let conf = kpts.series(k, \.conf)
            let valid = conf.map { $0 >= minConf }
            guard valid.filter({ $0 }).count >= 2 else { continue }
            var x = fillGaps(kpts.series(k, \.x), valid: valid)
            var y = fillGaps(kpts.series(k, \.y), valid: valid)
            (x, y) = removeSpikes(x: x, y: y)
            let xs = lowpass(x, fps: fps, cutoffHz: cutoffHz)
            let ys = lowpass(y, fps: fps, cutoffHz: cutoffHz)
            let cs = T >= 5 ? NaNMath.medianFilter(conf, size: 5) : conf
            for t in 0..<T {
                out[t][k] = Keypoint(x: xs[t], y: ys[t], conf: cs[t])
            }
        }
        return out
    }
}
