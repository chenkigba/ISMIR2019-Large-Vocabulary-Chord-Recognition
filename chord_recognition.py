from .chordnet_ismir_naive import ChordNet, chord_limit, ChordNetCNN
from .mir.nn.train import NetworkInterface
from .extractors.cqt import CQTV2, SimpleChordToID
from .mir import io, DataEntry
from .extractors.xhmm_ismir import XHMMDecoder
import numpy as np
from .io_new.chordlab_io import ChordLabIO
from .settings import DEFAULT_SR, DEFAULT_HOP_LENGTH
import sys
import os
import torch

MODEL_NAMES = [
    "joint_chord_net_ismir_naive_v1.0_reweight(0.0,10.0)_s%d.best" % i for i in range(5)
]


def chord_recognition(audio_path, lab_path, chord_dict_name="submission"):
    # Use log-prob decoding; disable caching (one-off processing)
    hmm = XHMMDecoder(
        template_file="data/%s_chord_list.txt" % chord_dict_name, log_input=True
    )
    # Empty name disables all extractors' cache
    entry = DataEntry()
    entry.prop.set("sr", DEFAULT_SR)
    entry.prop.set("hop_length", DEFAULT_HOP_LENGTH)
    entry.append_file(audio_path, io.MusicIO, "music")
    # Explicitly disable cache for this extractor call
    entry.append_extractor(CQTV2, "cqt", cache_enabled=False)
    # Prepare input tensor once and reuse across ensemble models
    cqt_np = entry.cqt  # triggers extraction (cached if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_input = torch.tensor(cqt_np, dtype=torch.float32, device=device)
    probs = []
    for model_name in MODEL_NAMES:
        net = NetworkInterface(
            ChordNet(None), model_name, load_checkpoint=False, inference_only=True
        )
        print("Inference: %s on %s" % (model_name, audio_path))
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

    probs = [log_mean_exp(h, axis=0) for h in log_probs]
    chordlab = hmm.decode_to_chordlab(entry, probs, False)
    entry.append_data(chordlab, ChordLabIO, "chord")
    entry.save("chord", lab_path)


def main():
    """Main entry point for the chord recognition command line tool."""
    if len(sys.argv) == 3:
        chord_recognition(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 4:
        chord_recognition(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print(
            "Usage: chord-recognition path_to_audio_file path_to_output_file [chord_dict=submission]"
        )
        print(
            "\tChord dict can be one of the following: full, ismir2017, submission, extended"
        )
        exit(0)


if __name__ == "__main__":
    main()
