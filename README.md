# whisper_subtitler
Generate transcriptions and subtitles using OpenAI whisper as a base model, stable-ts/whisperx as a timestamp stabilizer using ASR models and pyannote/nemo models in order to identify different speakers.

## Current pipeline
1. Read and preprocess the input file (video or audio)
2. Load whisper + timestamp stabilizer and inference with the input file
3. Load diarization model and run model (if selected by the user) with the input file
4. Form final results with post processing (fix output by punctuation, etc.)
5. Output results

## TODO
* ~~First approach with stable-ts whisper and pyannote~~
* ~~User interface using streamlit~~
* Alternative pipeline with whisperx and pyannote
* Alternative pipeline with whisperx (timestamp stabilizer + custom diarization)
* Include audio processing (Silero VAD, audio enhancer with espnet or similar)
* Alternative pipeline with Nvidia NEMO
* Alternative pipeline with stable-ts/whisperx and speech separation (instead of diarization)
* Include translation to the pipeline
