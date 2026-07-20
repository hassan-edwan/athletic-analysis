// NumPy-equivalent numerics with identical NaN semantics, so the Swift port
// can mirror the Python core line for line. Everything here is scalar Swift on
// [Double] — no Accelerate — to stay buildable on Windows/Linux.

import Foundation

enum NaNMath {
    /// numpy.nanmedian over the finite values; NaN when none.
    static func nanMedian(_ values: [Double]) -> Double {
        let finite = values.filter { $0.isFinite }.sorted()
        guard !finite.isEmpty else { return .nan }
        let n = finite.count
        return n % 2 == 1 ? finite[n / 2]
                          : 0.5 * (finite[n / 2 - 1] + finite[n / 2])
    }

    /// numpy.nanmean; NaN when no finite values.
    static func nanMean(_ values: [Double]) -> Double {
        var sum = 0.0
        var count = 0
        for v in values where v.isFinite {
            sum += v
            count += 1
        }
        return count > 0 ? sum / Double(count) : .nan
    }

    static func nanMax(_ values: [Double]) -> Double {
        var best = -Double.infinity
        var any = false
        for v in values where v.isFinite {
            best = Swift.max(best, v)
            any = true
        }
        return any ? best : .nan
    }

    /// numpy.nanpercentile with the default linear interpolation method.
    static func nanPercentile(_ values: [Double], _ q: Double) -> Double {
        let finite = values.filter { $0.isFinite }.sorted()
        guard !finite.isEmpty else { return .nan }
        if finite.count == 1 { return finite[0] }
        let pos = q / 100.0 * Double(finite.count - 1)
        let lower = Int(pos.rounded(.down))
        let upper = Swift.min(lower + 1, finite.count - 1)
        let frac = pos - Double(lower)
        return finite[lower] * (1 - frac) + finite[upper] * frac
    }

    /// numpy.gradient for 1-D uniform spacing: central differences inside,
    /// one-sided at the edges. NaNs propagate exactly as in NumPy.
    static func gradient(_ x: [Double]) -> [Double] {
        let n = x.count
        guard n >= 2 else { return x.map { _ in 0.0 } }
        var out = [Double](repeating: 0, count: n)
        out[0] = x[1] - x[0]
        out[n - 1] = x[n - 1] - x[n - 2]
        for i in 1..<(n - 1) {
            out[i] = (x[i + 1] - x[i - 1]) / 2.0
        }
        return out
    }

    /// numpy.interp(xq, xp, fp): xp strictly increasing; clamps to end values.
    static func interp(_ xq: [Double], xp: [Double], fp: [Double]) -> [Double] {
        precondition(xp.count == fp.count && !xp.isEmpty)
        return xq.map { x in
            if x <= xp[0] { return fp[0] }
            if x >= xp[xp.count - 1] { return fp[fp.count - 1] }
            // Binary search for the bracketing interval.
            var lo = 0
            var hi = xp.count - 1
            while hi - lo > 1 {
                let mid = (lo + hi) / 2
                if xp[mid] <= x { lo = mid } else { hi = mid }
            }
            let t = (x - xp[lo]) / (xp[hi] - xp[lo])
            return fp[lo] + t * (fp[hi] - fp[lo])
        }
    }

    /// Centered moving average that ignores NaNs (port of velocity.rolling_nanmean).
    static func rollingNanMean(_ x: [Double], window: Int) -> [Double] {
        let n = x.count
        guard n > 0, window > 1 else { return x }
        let half = window / 2
        var out = [Double](repeating: .nan, count: n)
        // O(n·window) is fine for clip-length series.
        for i in 0..<n {
            let lo = Swift.max(i - half, 0)
            let hi = Swift.min(i + half, n - 1)
            var sum = 0.0
            var count = 0
            for j in lo...hi where x[j].isFinite {
                sum += x[j]
                count += 1
            }
            if count > 0 { out[i] = sum / Double(count) }
        }
        return out
    }

    /// scipy.ndimage.median_filter(size:) in 1-D with the default 'reflect'
    /// border mode ((d c b a | a b c d …)).
    static func medianFilter(_ x: [Double], size: Int) -> [Double] {
        let n = x.count
        guard n > 0, size > 1 else { return x }
        let padLeft = size / 2
        let padRight = size - padLeft - 1
        var padded = [Double]()
        padded.reserveCapacity(n + padLeft + padRight)
        for i in stride(from: padLeft - 1, through: 0, by: -1) {
            padded.append(x[Swift.min(i, n - 1)])
        }
        padded.append(contentsOf: x)
        for i in 0..<padRight {
            padded.append(x[Swift.max(0, n - 1 - i)])
        }
        var out = [Double](repeating: 0, count: n)
        for i in 0..<n {
            let window = Array(padded[i..<(i + size)]).sorted()
            out[i] = size % 2 == 1 ? window[size / 2]
                                   : 0.5 * (window[size / 2 - 1] + window[size / 2])
        }
        return out
    }
}
