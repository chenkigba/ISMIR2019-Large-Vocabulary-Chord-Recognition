from .chordnet_ismir_naive import ChordNet
from .mir.nn.train import NetworkInterface
from .extractors.cqt import CQTV2
from .mir import io, DataEntry
from .extractors.xhmm_ismir import XHMMDecoder
import numpy as np
from .io_new.chordlab_io import ChordLabIO
from .settings import DEFAULT_SR, DEFAULT_HOP_LENGTH
from pathlib import Path
import sys
import time
import argparse
import torch
import librosa


MODEL_NAMES = [
    "joint_chord_net_ismir_naive_v1.0_reweight(0.0,10.0)_s%d.best" % i for i in range(5)
]


class StageTimer:
    """Minimal stage timer with optional printing and summary.

    Usage:
        timer = StageTimer(enabled=True)
        with timer.stage("Compute CQT"):
            ...
        timer.print_summary()
    """

    def __init__(self, enabled=False):
        self.enabled = enabled
        self.records = []  # list of (name, seconds)

    def stage(self, name, sync=None):
        class _Ctx:
            def __init__(self, outer):
                self.outer = outer
                self.name = name
                self.sync = sync
                self.t0 = None

            def __enter__(self):
                if callable(self.sync):
                    try:
                        self.sync()
                    except Exception:
                        pass
                self.t0 = time.perf_counter()
                return self

            def __exit__(self, exc_type, exc, tb):
                if callable(self.sync):
                    try:
                        self.sync()
                    except Exception:
                        pass
                dt = time.perf_counter() - self.t0
                self.outer.records.append((self.name, dt))
                if self.outer.enabled:
                    print(f"[time] {self.name}: {dt:.3f}s")
                # Do not suppress exceptions
                return False

        return _Ctx(self)

    def print_summary(self):
        if not self.enabled or not self.records:
            return
        totals = {}
        for name, dt in self.records:
            totals[name] = totals.get(name, 0.0) + dt
        total_time = sum(totals.values())
        print("[time] --- Summary ---")
        for name, dt in sorted(totals.items(), key=lambda x: x[1], reverse=True):
            print(f"[time] {name:<22} {dt:6.3f}s")
        print(f"[time] Total{'':<18} {total_time:6.3f}s")


def chord_recognition(
    audio_path, lab_path, chord_dict_name="submission", verbose=False
):
    # Use log-prob decoding; disable caching (one-off processing)
    timer = StageTimer(enabled=verbose)

    with timer.stage("Init HMM"):
        hmm = XHMMDecoder(
            template_file=str(
                Path(__file__).parent / "data" / f"{chord_dict_name}_chord_list.txt"
            ),
            log_input=True,
        )
    # Empty name disables all extractors' cache
    with timer.stage("Prepare entry"):
        entry = DataEntry()
        entry.prop.set("sr", DEFAULT_SR)
        entry.prop.set("hop_length", DEFAULT_HOP_LENGTH)
        entry.append_file(audio_path, io.MusicIO, "music")
        # Explicitly disable cache for this extractor call
        entry.append_extractor(CQTV2, "cqt", cache_enabled=False)

    # 先单独计时音频加载（解码）；随后再计时CQT计算
    with timer.stage("Load audio"):
        _ = entry.music

    # 计算CQT（混合CQT），与音频加载分开计时，便于定位瓶颈
    with timer.stage("Compute CQT"):
        cqt_np = entry.cqt  # triggers extraction (cached if available)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with timer.stage("Prepare tensor"):
        x_input = torch.tensor(cqt_np, dtype=torch.float32, device=device)

    probs = []

    # Function to sync CUDA for accurate timing, no-op on CPU
    def _sync_cuda():
        if device.type == "cuda":
            torch.cuda.synchronize()

    for model_name in MODEL_NAMES:
        with timer.stage(f"Init model: {model_name}"):
            net = NetworkInterface(
                ChordNet(None), model_name, load_checkpoint=False, inference_only=True
            )
        print("Inference: %s on %s" % (model_name, audio_path))
        with timer.stage(f"Inference: {model_name}", sync=_sync_cuda):
            probs.append(net.inference(x_input))
    # Average log-probs correctly: log-mean-exp across models
    # probs is a list of tuples (6 heads) with log-probs
    log_probs = [np.stack([p[i] for p in probs], axis=0) for i in range(len(probs[0]))]

    # log-mean-exp: log(sum(exp(logp)) / N) = logsumexp(logp) - log(N)
    def log_mean_exp(arr, axis=0):
        m = np.max(arr, axis=axis, keepdims=True)
        return (m + np.log(np.mean(np.exp(arr - m), axis=axis, keepdims=True))).squeeze(
            axis
        )

    with timer.stage("Ensemble log-mean-exp"):
        probs = [log_mean_exp(h, axis=0) for h in log_probs]
    with timer.stage("HMM decode"):
        chordlab = hmm.decode_to_chordlab(entry, probs, False)
    with timer.stage("Save output"):
        entry.append_data(chordlab, ChordLabIO, "chord")


def chord_recognition_from_memory(audio, sr, lab_path, chord_dict_name="submission", verbose=False, name="in-memory"):
    """基于内存中的音频数据运行和弦识别并保存到 .lab 文件。

    参数:
    - audio: numpy.ndarray, 1D 或 2D(stereo) 波形
    - sr: int, audio 的采样率
    - lab_path: 输出 .lab 路径
    - chord_dict_name: 和弦词典，默认 submission
    - verbose: 是否打印阶段耗时
    - name: 可选的音频标识名（用于日志）
    """
    timer = StageTimer(enabled=verbose)

    # 预处理：转单声道、类型与重采样
    with timer.stage("Prepare audio"):
        y = audio
        # 转 float32
        if y.dtype != np.float32:
            y = y.astype(np.float32, copy=False)
        # 立体声 -> 单声道
        if y.ndim == 2:
            # 支持 (2, N) 或 (N, 2)
            if y.shape[0] == 2 and y.shape[1] != 2:
                y = librosa.to_mono(y)
            else:
                y = librosa.to_mono(y.T) if y.shape[1] == 2 else y
        # 重采样到模型默认采样率（与特征参数一致）
        if sr != DEFAULT_SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=DEFAULT_SR)
            sr = DEFAULT_SR

    with timer.stage("Init HMM"):
        hmm = XHMMDecoder(
            template_file=str(
                Path(__file__).parent / "data" / f"{chord_dict_name}_chord_list.txt"
            ),
            log_input=True,
        )

    with timer.stage("Prepare entry"):
        entry = DataEntry()
        entry.prop.set("sr", DEFAULT_SR)
        entry.prop.set("hop_length", DEFAULT_HOP_LENGTH)
        # 直接注入内存中的波形
        entry.append_data(y, io.MusicIO, "music")
        entry.append_extractor(CQTV2, "cqt", cache_enabled=False)

    # 访问一次触发潜在的惰性流程（对 DataProxy 为 O(1)）
    with timer.stage("Load audio"):
        _ = entry.music

    with timer.stage("Compute CQT"):
        cqt_np = entry.cqt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with timer.stage("Prepare tensor"):
        x_input = torch.tensor(cqt_np, dtype=torch.float32, device=device)

    probs = []

    def _sync_cuda():
        if device.type == "cuda":
            torch.cuda.synchronize()

    for model_name in MODEL_NAMES:
        with timer.stage(f"Init model: {model_name}"):
            net = NetworkInterface(
                ChordNet(None), model_name, load_checkpoint=False, inference_only=True
            )
        print("Inference: %s on %s" % (model_name, name))
        with timer.stage(f"Inference: {model_name}", sync=_sync_cuda):
            probs.append(net.inference(x_input))

    log_probs = [np.stack([p[i] for p in probs], axis=0) for i in range(len(probs[0]))]

    def log_mean_exp(arr, axis=0):
        m = np.max(arr, axis=axis, keepdims=True)
        return (m + np.log(np.mean(np.exp(arr - m), axis=axis, keepdims=True))).squeeze(axis)

    with timer.stage("Ensemble log-mean-exp"):
        probs = [log_mean_exp(h, axis=0) for h in log_probs]
    with timer.stage("HMM decode"):
        chordlab = hmm.decode_to_chordlab(entry, probs, False)
    with timer.stage("Save output"):
        entry.append_data(chordlab, ChordLabIO, "chord")
        entry.save("chord", lab_path)

    timer.print_summary()


def main():
    """Main entry point for the chord recognition command line tool."""
    parser = argparse.ArgumentParser(
        prog="chord-recognition",
        description="Run chord recognition on an audio file and output a .lab file.",
    )
    parser.add_argument("audio_path", help="Path to input audio file")
    parser.add_argument("lab_path", help="Path to output chord lab file")
    parser.add_argument(
        "chord_dict",
        nargs="?",
        default="submission",
        choices=["full", "ismir2017", "submission", "extended"],
        help="Chord dictionary name (default: submission)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print per-stage timing and summary",
    )
    args = parser.parse_args(sys.argv[1:])

    chord_recognition(
        args.audio_path, args.lab_path, args.chord_dict, verbose=args.verbose
    )


if __name__ == "__main__":
    main()
