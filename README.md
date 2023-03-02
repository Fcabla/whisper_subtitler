# whisper_subtitler
Generate transcriptions and subtitles using OpenAI whisper as a base model, [stable-ts](https://github.com/jianfch/stable-ts) or [whisperx](https://github.com/m-bain/whisperX) as a timestamp stabilizer using ASR models and [pyannote](https://github.com/pyannote/pyannote-audio) / nemo models in order to identify different speakers.

<video width="320" height="240" controls>
  <source src="docs/demo.mp4" type="video/mp4">
</video>


## Current pipeline (transcriber_wx | whisperx)
1. Read and preprocess the input file (video or audio)
2. Load models: pyannote Voice Activity Detection (VAD), pyannote diarization, whisper, align-model (wav2vec2.0)
3. Transcribe input using VAD and whisper
4. Run alignment model to stabilize the timestamps from whisper
5. Perform diarization
5. Form final results and output.

## Previous pipeline (transcriber_sw | stable-whisper)
1. Read and preprocess the input file (video or audio)
2. Load whisper + timestamp stabilizer and inference with the input file
3. Load diarization model and run model (if selected by the user) with the input file
4. Form final results with post processing (fix output by punctuation, etc.)
5. Output results

## Limitations
* The main limitations besides inference time (depending on the selected model) its the overlapping speakers. When 2 or more speakers speaks at the same time, whisper just outputs the transcription of one speaker. Also, the diarization models can output sequences that overlapp bethween each other, but whisper will just output one token for a given timestamp. You can end in a situation where you have one token and multiple possible speakers.
* Untested in different languages, but should work.
* Untested in longer audio/video files. Maybe would be a good idea to split files into smaller chunks if this is a problem.


## TODO
* ~~First approach with stable-ts whisper and pyannote~~
* ~~User interface using streamlit~~
* ~~Alternative pipeline with whisperx and pyannote~~
* ~~Alternative pipeline with whisperx (timestamp stabilizer + custom diarization)~~
* ~~Pipeline with whisperx as timestamp stabilizer + VAD + diarization~~
* Include audio processing (Silero VAD, audio enhancer with espnet or similar)
* Try pipeline with Nvidia NEMO
* Implement speech separation
* Alternative pipeline with stable-ts/whisperx and speech separation (instead of diarization)
* Include translation to the pipeline
* ~~Add demo video~~