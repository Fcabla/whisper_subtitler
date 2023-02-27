"""
First pipeline proposed. Stable ts + whisper + pydub + align timestamps from transcription with diarization.
"""
# Private file with my hugging faces tokens
import utils.token_keys as tk
import utils.audio_procesing as ap
import utils.representation_classes as rc
import os, datetime, shutil, time
from pyannote.audio import Pipeline
import ffmpeg
from pyannote.audio import Inference, Model, Pipeline
import numpy as np
import pandas as pd
import torch
from whisperx import load_model, load_align_model, transcribe_with_vad, transcribe_with_vad_parallel, transcribe #, align, get_trellis, backtrack, merge_repeats, merge_words
from whisperx.alignment import align
from whisperx.diarize import assign_word_speakers
from whisperx.utils import write_txt, write_srt, format_timestamp
from typing import TextIO, Iterator

##################################
### HYPERPARAMETERS AND CONFIG ###
##################################
DEVICE = "cuda" #torch.device('cuda' if torch.cuda.is_available() else 'cpu') #"cuda" if torch.cuda.is_available() else "cpu"
#model_name = "medium" #tiny.en,tiny,base.en,base,small.en,small,medium.en,medium,large-v1,large-v2,large  (argumento)
MODEL_DIR = "~/.cache/whisper" #None#the path to save model files; uses ~/.cache/whisper by default (opcional)
ALIGN_MODEL = 'WAV2VEC2_ASR_LARGE_LV60K_960H' #"Name of phoneme-level ASR model to do alignment"
ALIGN_EXTEND = 2 #"Seconds before and after to extend the whisper segments for alignment"
ALIGN_FROM_PREV = True #"Whether to clip the alignment start time of current segment to the end time of the last aligned word of the previous segment
INTERPOLATE_METHOD = "nearest" #["nearest", "linear", "ignore"], help="For word .srt, method to assign timestamps to non-aligned words, or merge them into neighbouring.")
TEMPERATURE = 0 #"temperature to use for sampling"
TEMPERATURE_INCREMENT_ON_FALLBACK = 0.2 #"temperature to increase when falling back when the decoding fails to meet either of the thresholds below"
THREADS = 0
#ALIGN_LANGUAGE = "en" # default to loading english if not specified (NOT USED)
HF_TOKEN = tk.hf_token
VAD_FILTER = True #"Whether to first perform VAD filtering to target only transcribe within VAD. Produces more accurate alignment + timestamp, requires more GPU memory & compute.")
PARALLEL_BS = -1 #Enable parallel transcribing if > 1

#diarize = True #"Apply diarization to assign speaker labels to each segment/word"
MIN_SPEAKERS = None
MAX_SPEAKERS = None

#################
### FUNCTIONS ###
#################
def write_srt(transcript: Iterator[dict], file: TextIO, spk_colors=None):
    """
    Write a transcript to a file in SRT format.

    Example usage:
        from pathlib import Path
        from whisper.utils import write_srt

        result = transcribe(model, audio_path, temperature=temperature, **args)

        # save SRT
        audio_basename = Path(audio_path).stem
        with open(Path(output_dir) / (audio_basename + ".srt"), "w", encoding="utf-8") as srt:
            write_srt(result["segments"], file=srt)
    """
    #spk_colors = {'SPEAKER_00':'white','SPEAKER_01':'yellow'}
    for i, segment in enumerate(transcript, start=1):
        # write srt lines
        
        text = f"{segment['text'].strip().replace('-->', '->')}"
        if spk_colors and 'speaker' in segment.keys():
            #f'<font color="{spk_colors[sentence.speaker]}">{text}</font>'
            text = f'<font color="{spk_colors[segment["speaker"]]}">{text}</font>'
        text += "\n"
        print(
            f"{i}\n"
            f"{format_timestamp(segment['start'], always_include_hours=True, decimal_marker=',')} --> "
            f"{format_timestamp(segment['end'], always_include_hours=True, decimal_marker=',')}\n"
            f"{text}",
            file=file,
            flush=True,
        )
def burn_subtitles(input_file, source_type, input_srt, output_video):
    if source_type == 'video':
        video = ffmpeg.input(input_file)
        audio = video.audio
        ffmpeg.concat(video.filter("subtitles", input_srt), audio, v=1, a=1).output(output_video).run()
    elif source_type == 'audio':
        audio_len = ffmpeg.probe(input_file)["format"]["duration"]
        video = ffmpeg.input("color=c=black:s=640x200", f="lavfi", t=audio_len)
        audio = ffmpeg.input(input_file)
        ffmpeg.concat(video.filter("subtitles", input_srt, force_style="Alignment=10,FontSize=40"), audio, v=1, a=1).output(output_video).run()
    #ffmpeg -i subtitles.srt subtitles.ass
    #ffmpeg -i mymovie.mp4 -vf ass=subtitles.ass mysubtitledmovie.mp4

################
### PIPELINE ###
################
def pipeline(original_file, source_type, source_lan, model_type, use_diarization, num_speakers, translate_to_english):

    # Start measure time
    start_t = time.time()
    # 1. Create output, output/temp, output/temp/diarizations directories
    output_folder = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_folder = os.path.join('output', output_folder)
    os.mkdir(output_folder)
    temp_folder = os.path.join(output_folder, 'temp')
    os.mkdir(temp_folder)
    # Copy original file into output

    # 2. Preprocess and Format the audio file into wav
    last_audiofile = os.path.join(temp_folder,'audio.wav')
    ap.file_to_wav(input_file=original_file, output_file=last_audiofile)
    # Add a 0.5 miliseconds spacer into the audio file, since diarization struggles at the start
    #ap.add_init_spacer(input_file=last_audiofile, output_file=last_audiofile)

    # 3. LOAD MODELS
    # 3.1 Load VAD model
    vad_pipeline = None
    if VAD_FILTER:
        if HF_TOKEN is None:
            print("Warning, no huggingface token used, needs to be saved in environment variable, otherwise will throw error loading VAD model...")
        vad_pipeline =  Model.from_pretrained("pyannote/segmentation", use_auth_token=HF_TOKEN)
        vad_pipeline = Inference(model=vad_pipeline, pre_aggregation_hook=lambda segmentation: segmentation, use_auth_token=HF_TOKEN, device=torch.device(DEVICE))
    
    # 3.2 Load diarization
    diarize_pipeline = None
    if use_diarization:
        if HF_TOKEN is None:
            print("Warning, no --hf_token used, needs to be saved in environment variable, otherwise will throw error loading diarization model...")
        diarize_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=HF_TOKEN)

    # 3.3 Config stuff
    if TEMPERATURE_INCREMENT_ON_FALLBACK is not None:
        temperature = tuple(np.arange(TEMPERATURE, 1.0 + 1e-6, TEMPERATURE_INCREMENT_ON_FALLBACK))
    else:
        temperature = [TEMPERATURE]
    
    if THREADS > 0:
        torch.set_num_threads(THREADS)

    whisper_task = 'transcribe'
    if translate_to_english and source_lan != 'en':
        whisper_task = 'translate'

    # 3.4 Load Whisper model
    model = load_model(model_type, device=DEVICE, download_root=MODEL_DIR)

    # 3.5 Load alignmodel
    #ALIGN_LANGUAGE = source_lan
    align_model, align_metadata = load_align_model(source_lan, DEVICE, model_name=ALIGN_MODEL)

    # 4. INFERENCE
    #for audio_path in args.pop("audio"):
    #audio_path = 'input/original.wav'
    # 4.1 Run transcription inference with/without VAD
    if VAD_FILTER:
        if PARALLEL_BS > 1:
            print("Performing VAD and parallel transcribing ...")
            #result = transcribe_with_vad_parallel(model, audio_path, vad_pipeline, temperature=temperature, batch_size=parallel_bs, **args)
            result = transcribe_with_vad_parallel(model, last_audiofile, vad_pipeline, temperature=temperature, batch_size=PARALLEL_BS, task=whisper_task)
        else:
            print("Performing VAD...")
            #result = transcribe_with_vad(model, audio_path, vad_pipeline, temperature=temperature, **args)
            result = transcribe_with_vad(model, last_audiofile, vad_pipeline, temperature=temperature, task=whisper_task)
    else:
        print("Performing transcription...")
        result = transcribe(model, last_audiofile, temperature=temperature)
    
    # 4.2 rerun loading alignment model if different lang 
    if result["language"] != align_metadata["language"]:
        # load new language
        print(f"New language found ({result['language']})! Previous was ({align_metadata['language']}), loading new alignment model for new language...")
        align_model, align_metadata = load_align_model(result["language"], DEVICE)

    # 4.3 Run alignment
    print("Performing alignment...")
    result_aligned = align(result["segments"], align_model, align_metadata, last_audiofile, DEVICE,
                            extend_duration=ALIGN_EXTEND, start_from_previous=ALIGN_FROM_PREV, interpolate_method=INTERPOLATE_METHOD)
    # name of the audio file
    audio_basename = os.path.basename(last_audiofile)

    # 4.4 Perform diarization
    if use_diarization:
        print("Performing diarization...")
        #diarize_segments = diarize_pipeline(last_audiofile, min_speakers=min_speakers, max_speakers=max_speakers)
        diarize_segments = diarize_pipeline(last_audiofile, num_speakers=num_speakers)
        diarize_df = pd.DataFrame(diarize_segments.itertracks(yield_label=True))
        diarize_df['start'] = diarize_df[0].apply(lambda x: x.start)
        diarize_df['end'] = diarize_df[0].apply(lambda x: x.end)
        # assumes each utterance is single speaker (needs fix)
        result_segments, word_segments = assign_word_speakers(diarize_df, result_aligned["segments"], fill_nearest=True)
        result_aligned["segments"] = result_segments
        result_aligned["word_segments"] = word_segments
    
    # 5. Results
    #print(result_aligned["segments"],result_aligned["word_segments"])

    # 5.1 Raw txt
    output_raw_text = os.path.join(output_folder, 'raw_transcription.txt')
    with open(output_raw_text, 'w', encoding='utf-8') as f:
        #f.write(raw_transcripted_text)
        write_txt(result_aligned["segments"], file=f)

    # 5.2 timestamped (diarized) txt
    transcripted_text = []
    with open(os.path.join(output_folder, 'transcription.txt'), 'w', encoding='utf-8') as f:
        for sentence in result_aligned["segments"]:
            if use_diarization:
                spk_text = f'{sentence["speaker"]}: {sentence["text"]}'
            else:
                spk_text = f'{sentence["text"]}'
            sent_start = round(sentence["start"], 3)
            sent_end = round(sentence["end"], 3)
            transcripted_text.append(f'[{sent_start}-{sent_end}] {spk_text}')
            f.write(spk_text)
            f.write('\n')
    print(f'Results stored in:{output_folder}/transcription.txt')

    # 5.3 SRT (optional: add colors)
    spk_colors = {'SPEAKER_00':'white','SPEAKER_01':'yellow'}
    output_srt = os.path.join(output_folder, 'transcription.srt')
    with open(output_srt, "w", encoding="utf-8") as srt:
        write_srt(result_aligned["segments"], file=srt, spk_colors=spk_colors)
        # srt word level
        #write_srt(result_aligned["word_segments"], file=srt)

    # 5.4 Burn subtittles in video
    output_video = os.path.join(output_folder, 'result.mp4')
    burn_subtitles(input_file=original_file, source_type=source_type, input_srt=output_srt, output_video=output_video)

    shutil.rmtree(temp_folder)

    # End measure time
    exec_time = time.time() - start_t
    print(f'Inference time with unk device: {exec_time}')

    result_object = {
        'transcripted_text': transcripted_text,
        'output_raw_text': output_raw_text,
        'output_srt': output_srt,
        'output_video': output_video,
        'exec_time': exec_time
    }

    return result_object
    """
    from whisperx.utils import write_vtt, write_tsv, write_ass
    output_dir = 't'
    output_type = 'all'
    # save TXT
    if output_type in ["txt", "all"]:
        with open(os.path.join(output_dir, audio_basename + ".txt"), "w", encoding="utf-8") as txt:
            write_txt(result_aligned["segments"], file=txt)

    # save VTT
    if output_type in ["vtt", "all"]:
        with open(os.path.join(output_dir, audio_basename + ".vtt"), "w", encoding="utf-8") as vtt:
            write_vtt(result_aligned["segments"], file=vtt)

    # save SRT
    if output_type in ["srt", "all"]:
        with open(os.path.join(output_dir, audio_basename + ".srt"), "w", encoding="utf-8") as srt:
            write_srt(result_aligned["segments"], file=srt)

    # save TSV
    if output_type in ["tsv", "all"]:
        with open(os.path.join(output_dir, audio_basename + ".tsv"), "w", encoding="utf-8") as srt:
            write_tsv(result_aligned["segments"], file=srt)

    # save SRT word-level
    if output_type in ["srt-word", "all"]:
        # save per-word SRT
        with open(os.path.join(output_dir, audio_basename + ".word.srt"), "w", encoding="utf-8") as srt:
            write_srt(result_aligned["word_segments"], file=srt)

    # save ASS
    if output_type in ["ass", "all"]:
        with open(os.path.join(output_dir, audio_basename + ".ass"), "w", encoding="utf-8") as ass:
            write_ass(result_aligned["segments"], file=ass)
    
    # # save ASS character-level
    if output_type in ["ass-char"]:
        with open(os.path.join(output_dir, audio_basename + ".char.ass"), "w", encoding="utf-8") as ass:
            write_ass(result_aligned["segments"], file=ass, resolution="char")

    # save word tsv
    if output_type in ["pickle"]:
        exp_fp = os.path.join(output_dir, audio_basename + ".pkl")
        pd.DataFrame(result_aligned["segments"]).to_pickle(exp_fp)

    # save word tsv
    if output_type in ["vad"]:
        exp_fp = os.path.join(output_dir, audio_basename + ".sad")
        wrd_segs = pd.concat([x["word-segments"] for x in result_aligned["segments"]])[['start','end']]
        wrd_segs.to_csv(exp_fp, sep='\t', header=None, index=False)
    """

if __name__ =='__main__':
    pipeline(original_file='input/original.mp4', source_type='video', source_lan = 'en', model_type='medium', use_diarization=True, num_speakers=2)
