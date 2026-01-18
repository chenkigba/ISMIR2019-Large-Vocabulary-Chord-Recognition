# Large-Vocabulary Chord Transcription via Chord Structure Decomposition

This is the official repo for the ISMIR 2019 paper [Large-Vocabulary Chord Transcription via Chord Structure Decomposition](https://archives.ismir.net/ismir2019/paper/000078.pdf).

Screenshot for audio chord recognition (visualized via [Sonic Visualizer](https://www.sonicvisualiser.org/)): 
![image](https://user-images.githubusercontent.com/13694510/230480932-6df239fc-bb0c-4dbf-9399-ab79b3d82587.png)

## Pretrained models

This repo contains the pretrained models that does not incorporate label reweighting. This model provides the best overall accuracy. However, if you want the pretrained models with label reweighting, you can download it here:

https://drive.google.com/drive/u/1/folders/1y5-zTFaBliymPe7uY2MZfUAsvPzwmGBL

## Chord recognition with pretrained model

After installing all the dependencies, run the following code:

```
python3 chord_recognition.py path_to_audio_file path_to_output_file [chord_dict]
```

For example,
```
python3 chord_recognition.py example.mp3 example_chord.lab
```

Here, ``chord_dict`` is an optional parameter that tells the HMM which chord dictionary to use for decoding. Potential options are:

* submission: the default value (recommended to use). It is the chord dictionary we used for ISMIR 2019 submission.
* ismir2017: the chord dictionary for MIREX competition.
* full: the list of all chords from MARL dataset. It is not tested and not recommended to use. 

You may also manually adjust the chord dictionary by editing them in the folder ``data/*_chord_list.txt``.

## Beat-Guided Chord Recognition

You can improve chord boundary alignment by providing beat information to the HMM decoder. This feature constrains chord transitions to occur at beat positions, which better reflects how music is structured.

### Python API

```python
from lvcr_ismir2019.chord_recognition import chord_recognition

# Without beat guidance (original behavior)
chord_recognition(audio_path, output_path, chord_dict="submission")

# With beat guidance (new feature)
import numpy as np
beats = np.array([0.5, 1.0, 1.5, 2.0, ...])      # beat times in seconds
downbeats = np.array([0.5, 2.5, 4.5, ...])       # downbeat times in seconds

chord_recognition(
    audio_path, output_path, chord_dict="submission",
    beats=beats, downbeats=downbeats
)
```

### How It Works

When beat information is provided:
1. The HMM decoder restricts chord transitions to occur only at beat positions
2. Downbeats (beat 1 of each bar) have the lowest transition penalty, making chord changes most likely there
3. Other beats have moderate transition penalties
4. Between beats, chord transitions are not allowed

This results in chord boundaries that align with the musical meter, which is desirable for applications like automatic accompaniment and music analysis.

### Integration with beat_this

You can use [beat_this](https://github.com/CPJKU/beat_this) to extract beats automatically:

```python
from beat_this.inference import Audio2Beats
from lvcr_ismir2019.chord_recognition import chord_recognition
import librosa

# Load audio
audio, sr = librosa.load("song.wav", sr=None)

# Extract beats
audio2beats = Audio2Beats(device="cuda", dbn=True)
beats, downbeats = audio2beats(audio, sr)

# Extract chords with beat guidance
chord_recognition(
    "song.wav", "chords.lab",
    beats=beats, downbeats=downbeats
)
```

## Training

First prepare the jams dataset in the following format:

```
    chord_data_1217/
        audio/
            TR6R91L11C8A40D710.mp3
            ...
        chordlab/
            TR6R91L11C8A40D710.lab
            ...
```

Then modify the value of JAM_DATASET_PATH in ``settings.py`` to the path of the dataset (e.g., ``some_path/chord_data_1217``)

Then run ``storage_creation.py`` to get h5 data files ``jams_xchord.h5d`` and ``jams_cqt.h5d``.

Then run ``chordnet_ismir_naive.py 0`` for training/testing on data split #0.

If you encounter with some errors with ``torch.bool()`` 
just follows the error message to add some ``bool()`` to index tensors.

## Testing

Run ``chordnet_ismir_naive_eval.py`` for testing.

Junyan
