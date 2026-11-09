// Bench for a native small.en teacher-forced scorer on a real iPhone, via whisper.cpp's C API (ggml, Metal or CPU).
// Per asked line: mel + encode once, then every candidate's token list scored under the decoder, one token per whisper_decode call:
//   MODE T  trie DFS = prefix sharing (re-use n_past at a branch point; later KV entries are overwritten). steps = trie nodes.
//   MODE N  naive = each candidate re-decoded from the prompt, no sharing. steps = sum of candidate lengths.
import Foundation
import whisper

struct Line: Decodable { let asked: String; let askedIndex: Int; let texts: [String]; let tokens: [[Int32]] }
struct Cands: Decodable { let prompt: [Int32]; let eot: Int32; let lines: [Line] }

func nowMs() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1e6 }

func footprintMB() -> Double {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
    }
    return kr == KERN_SUCCESS ? Double(info.phys_footprint) / 1048576 : -1
}

func logp(_ p: UnsafePointer<Float>, nVocab: Int, tok: Int32) -> Double {
    var m = -Float.infinity
    for i in 0..<nVocab { m = max(m, p[i]) }
    var s: Double = 0
    for i in 0..<nVocab { s += exp(Double(p[i] - m)) }
    return Double(p[Int(tok)] - m) - log(s)
}

final class Node { var kids: [Int32: Node] = [:]; var leaf: [Int] = [] }

final class Bench {
    var ctx: OpaquePointer
    let nVocab: Int
    let nThreads: Int32 = 4
    var peakMB: Double = 0
    init?(modelPath: String, useGpu: Bool) {
        var cp = whisper_context_default_params()
        cp.use_gpu = useGpu
        guard let c = whisper_init_from_file_with_params(modelPath, cp) else { return nil }
        ctx = c
        nVocab = Int(whisper_n_vocab(c))
    }
    deinit { whisper_free(ctx) }
    func samplePeak() { peakMB = max(peakMB, footprintMB()) }

    func lastRow() -> [Float] { Array(UnsafeBufferPointer(start: whisper_get_logits(ctx)!, count: nVocab)) }

    func run(line: Line, cands: Cands, pcm: [Float], mode: String, gpu: Bool) -> [String: Any] {
        let t0 = nowMs()
        _ = whisper_pcm_to_mel(ctx, pcm, Int32(pcm.count), nThreads)
        _ = whisper_encode(ctx, 0, nThreads)
        let encodeMs = nowMs() - t0
        samplePeak()
        var score = [Double](repeating: 0, count: line.tokens.count)
        var calls = 0, steps = 0
        let d0 = nowMs()
        var prompt = cands.prompt
        _ = whisper_decode(ctx, &prompt, Int32(prompt.count), 0, nThreads)
        calls += 1
        let first = lastRow()
        let pl = prompt.count
        if mode == "T" {
            let root = Node()
            for (ci, toks) in line.tokens.enumerated() {
                var n = root
                for t in toks + [cands.eot] {
                    if n.kids[t] == nil { n.kids[t] = Node() }
                    n = n.kids[t]!
                }
                n.leaf.append(ci)
            }
            func dfs(_ n: Node, pos: Int, row: [Float], acc: Double) {
                for (tok, kid) in n.kids {
                    let a = acc + row.withUnsafeBufferPointer { logp($0.baseAddress!, nVocab: nVocab, tok: tok) }
                    for ci in kid.leaf { score[ci] = a }
                    if kid.kids.isEmpty { continue }
                    var t = [tok]
                    _ = whisper_decode(ctx, &t, 1, Int32(pos), nThreads)
                    calls += 1; steps += 1
                    dfs(kid, pos: pos + 1, row: lastRow(), acc: a)
                }
            }
            dfs(root, pos: pl, row: first, acc: 0)
        } else {
            for (ci, toks) in line.tokens.enumerated() {
                var row = first
                var acc = 0.0
                var pos = pl
                let seq = toks + [cands.eot]
                for (k, tok) in seq.enumerated() {
                    acc += row.withUnsafeBufferPointer { logp($0.baseAddress!, nVocab: nVocab, tok: tok) }
                    if k == seq.count - 1 { break }
                    var t = [tok]
                    _ = whisper_decode(ctx, &t, 1, Int32(pos), nThreads)
                    calls += 1; steps += 1; pos += 1
                    row = lastRow()
                }
                score[ci] = acc
            }
        }
        let decodeMs = nowMs() - d0
        samplePeak()
        let alt = score.enumerated().filter { $0.offset != line.askedIndex }.map { $0.element }.max() ?? -Double.infinity
        return ["asked": line.asked, "mode": mode, "gpu": gpu, "encodeMs": encodeMs, "decodeMs": decodeMs, "calls": calls, "steps": steps,
                "cands": line.tokens.count, "margin": score[line.askedIndex] - alt, "footMB": footprintMB(), "peakMB": peakMB,
                "thermal": ProcessInfo.processInfo.thermalState.rawValue]
    }
}
