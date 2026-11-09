import SwiftUI
import Foundation

func loadWav(_ url: URL) -> [Float] {
    let d = try! Data(contentsOf: url)
    var off = 12
    while off + 8 <= d.count {   // walk the RIFF chunks to "data"
        let id = String(decoding: d[off..<off + 4], as: UTF8.self)
        let sz = Int(d[off + 4]) | Int(d[off + 5]) << 8 | Int(d[off + 6]) << 16 | Int(d[off + 7]) << 24
        if id == "data" { off += 8; break }
        off += 8 + sz
    }
    var out = [Float]()
    var i = off
    while i + 1 < d.count {
        out.append(Float(Int16(bitPattern: UInt16(d[i]) | UInt16(d[i + 1]) << 8)) / 32768)
        i += 2
    }
    return out
}

@MainActor final class Model: ObservableObject {
    @Published var rows: [String] = ["starting"]
    var started = false
    func add(_ s: String) { rows.append(s); print("BENCH " + s) }
    func start() {
        if started { return }
        started = true
        Task.detached(priority: .userInitiated) { [self] in
            let b = Bundle.main
            let modelPath = b.path(forResource: "ggml-small.en", ofType: "bin")!
            let cands = try! JSONDecoder().decode(Cands.self, from: Data(contentsOf: b.url(forResource: "candidates", withExtension: "json")!))
            let pcm = loadWav(b.url(forResource: "jfk", withExtension: "wav")!)
            await add("INFO cpus=\(ProcessInfo.processInfo.processorCount) ram=\(ProcessInfo.processInfo.physicalMemory / 1048576)MB pcm=\(pcm.count) lines=\(cands.lines.count) base=\(Int(footprintMB()))MB")
            for gpu in [true, false] {
                let t0 = nowMs()
                guard let bench = Bench(modelPath: modelPath, useGpu: gpu) else { await add("ERR load gpu=\(gpu)"); continue }
                await add("LOAD gpu=\(gpu) ms=\(Int(nowMs() - t0)) footMB=\(Int(footprintMB()))")
                for mode in ["T", "N"] {
                    for line in cands.lines {
                        let r = bench.run(line: line, cands: cands, pcm: pcm, mode: mode, gpu: gpu)
                        let g = gpu ? "gpu" : "cpu"
                        let m = String(format: "%.2f", r["margin"] as! Double)
                        await add("R|\(mode)|\(g)|\(line.asked)|enc=\(Int(r["encodeMs"] as! Double))|dec=\(Int(r["decodeMs"] as! Double))|calls=\(r["calls"]!)|steps=\(r["steps"]!)|cands=\(r["cands"]!)|margin=\(m)|foot=\(Int(r["footMB"] as! Double))|peak=\(Int(r["peakMB"] as! Double))|th=\(r["thermal"]!)")
                    }
                }
            }
            await add("DONE")
        }
    }
}

@main struct WhisperBenchApp: App {
    @StateObject var m = Model()
    var body: some Scene {
        WindowGroup {
            ScrollView {
                VStack(alignment: .leading, spacing: 2) {
                    ForEach(Array(m.rows.enumerated()), id: \.offset) { item in
                        Text(item.element).font(.system(size: 9, design: .monospaced)).accessibilityLabel(item.element)
                    }
                }.padding(6)
            }.onAppear { m.start() }
        }
    }
}
