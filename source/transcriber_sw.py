"""
First pipeline proposed. Stable ts + whisper + pydub + align timestamps from transcription with diarization.
"""
# Private file with my hugging faces tokens
import utils.token_keys as tk
import utils.audio_procesing as ap
import utils.representation_classes as rc
import os, datetime, shutil
from pysrt import SubRipFile, SubRipTime, SubRipItem
import time
from pyannote.audio import Pipeline
import stable_whisper as sw
import ffmpeg

DEVICE = "cuda" #torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def diarizate_audio_file(diarization, input_file, nspeakers=0, store_intermediate_results_folder=None):
    # Running pyannote.audio to generate the diarizations.
    DEMO_FILE = {'uri': 'blabla', 'audio': input_file}
    if nspeakers == 0:
        # guess number of speakersW
        dz = diarization(DEMO_FILE) 
    else:
        dz = diarization(DEMO_FILE, num_speakers=nspeakers) 
    result = []
    for turn, _, speaker in dz.itertracks(yield_label=True):
        result.append({
            'start': turn.start,
            'end': turn.end,
            'duration': turn.end - turn.start,
            'speaker': speaker
        })
        print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
    
    if store_intermediate_results_folder:
        if store_intermediate_results_folder[-1] != '/':
            store_intermediate_results_folder += '/'
        with open(f"{store_intermediate_results_folder}diarization.txt", "w") as text_file:
            text_file.write(str(dz))
        with open(f"{store_intermediate_results_folder}diarization.rttm", "w") as rttm:
            dz.write_rttm(rttm)
    return result

def perform_transcription(transcriber, input_file, lang='unk'):
    if lang == 'unk':
        lang = None
    #results = transcriber.transcribe(AUDIO_FILE, fp16=True, language='en', suppress_silence=False, ts_num=16)
    results = transcriber.transcribe(input_file, fp16=False, language=lang)
    
    #from stable_whisper import stabilize_timestamps
    #stab_segments = stabilize_timestamps(results, top_focus=True)
    #print(stab_segments)
    # revisit this stabilize method
    #results = sw.stabilize_timestamps(results, top_focus=True)
    # Store words
    id_counter = 0
    words =  []
    for result in results['segments']:
        for elem in result['whole_word_timestamps']:
            elem['word'] = elem['word'].strip()
            new_word = rc.Trans_word(token=elem['word'], start=elem['timestamp'], id_order=id_counter)
            words.append(new_word)
            id_counter = id_counter + 1
    return words, results

def speaker_recognition(results_transcriptions, dz, unk_strat='closest'):
    if unk_strat not in ['closest', 'last']:
        print('unk_strat not recognized')
        exit()
    # Calculate speaker
    for word_id in range(len(results_transcriptions)):
        word = results_transcriptions[word_id]
        candidates = [elem for elem in dz if elem['start'] <= word.start <= elem['end']]
        if candidates:
            speaker = min(candidates, key=lambda x:x['duration'])
            word.set_speaker(speaker['speaker'])
        else:
            if unk_strat=='closest':
                unk_candidate = None
                for elem in dz:
                    delta_lower = abs(word.start - elem['start'])
                    delta_upper = abs(word.start - elem['end'])
                    elem['delta'] = min(delta_lower, delta_upper)
                    if unk_candidate:
                        if elem['delta'] < unk_candidate['delta']:
                            unk_candidate = elem
                    else:
                        unk_candidate = elem
                print(f"for the unkown elem: {word.token}[{word.start}] -> speaker:{unk_candidate['speaker']}[{unk_candidate['start']}:{unk_candidate['end']}]")
            elif unk_strat=='last':
                if word_id > 0:
                    unk_candidate = results_transcriptions[word_id-1].todict()
                else:
                    print('error')
                    exit()
                print(f"for the unkown elem: {word.token}[{word.start}] -> speaker:{unk_candidate['speaker']}[{unk_candidate['start']}:{unk_candidate['end']}]")
            word.set_speaker(unk_candidate['speaker'])
            
    return results_transcriptions

def paint_results(words):
    spk_colors = {0:rc.bcolors.OKBLUE,1:rc.bcolors.OKGREEN,2:rc.bcolors.WARNING,3:rc.bcolors.HEADER}
    for word in words:
        if word.speaker[-1] == 'UNK':
            print(f"{rc.bcolors.FAIL}{word.token}{rc.bcolors.ENDC}", end = '')
        else:
            color = spk_colors[int(word.speaker[-1])]
            print(f"{color}{word.token}{rc.bcolors.ENDC}", end = '')
        print(" ", end = '')

def build_result_transcription(words):
    # Sort by id
    words.sort(key=lambda x: x.id_order, reverse=False)
    results = []
    sentence_id = 0
    words_sentence = [words[0]]
    for word_index in range(1, len(words)):
        if words[word_index].speaker != words_sentence[-1].speaker:
            # if actual word ends with . or ? and starts with lower character then probably the speaker is wrong
            if words[word_index].token[0].islower() and words[word_index].token[-1] in ['.','?']:
                print(words[word_index])
                words[word_index].speaker = words_sentence[-1].speaker
                print(words[word_index])
            # if actual word is between different speakers and does not begin with caps, probably wrong speaker
            elif word_index < len(words) and words[word_index].token[0].islower() and words[word_index+1].token[0].islower():
                print(words[word_index])
                words[word_index].speaker = words_sentence[-1].speaker
                print(words[word_index])
            else:
                results.append(rc.Trans_sentence(words=words_sentence, speaker=words_sentence[0].speaker, sentence_id=sentence_id))
                sentence_id += 1
                words_sentence = []
        words_sentence.append(words[word_index])
    results.append(rc.Trans_sentence(words=words_sentence, speaker=words_sentence[-1].speaker, sentence_id=sentence_id))
    return(results)

def generate_srt(sentences, output_file, ann_offset=2000):
    spk_colors = {'SPEAKER_00':'white','SPEAKER_01':'yellow'}
    subs = SubRipFile()
    sentences = rc.get_splitted_sentence(sentences)
    #sentences = rc.smooth_timestamps(sentences)
    for sentence in sentences:
        text = sentence.text
        subitem = SubRipItem()
        subitem.start = SubRipTime(seconds=sentence.start_sent)
        subitem.end = SubRipTime(seconds=sentence.end_sent)
        subitem.index = sentence.sentence_id
        if sentence.speaker:
            subitem.text = f'<font color="{spk_colors[sentence.speaker]}">{text}</font>'
        else:
            subitem.text = text
        subs.append(subitem)
    
    # Fix offset included from pyannote
    if ann_offset:
        subs.shift(milliseconds=-ann_offset)

    subs.save(output_file, encoding='utf-8')
    return subs

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


def pipeline(original_file, source_type, source_lan, model_type, use_diarization, num_speakers):
    # Start measure time
    start_t = time.time()

    # 1. Create output, output/temp, output/temp/diarizations directories
    output_folder = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_folder = os.path.join('output', output_folder)
    os.mkdir(output_folder)
    temp_folder = os.path.join(output_folder, 'temp')
    os.mkdir(temp_folder)
    # Copy original file into output
    print(1)
    # 2. Preprocess and Format the audio file into wav
    last_audiofile = os.path.join(temp_folder,'audio.wav')
    ap.file_to_wav(input_file=original_file, output_file=last_audiofile)
    # Add a 0.5 miliseconds spacer into the audio file, since diarization struggles at the start
    ap.add_init_spacer(input_file=last_audiofile, output_file=last_audiofile)
    print(2)
    # 3. Load whisper model from stable whisper and inference
    transcriber = sw.load_model(model_type, device=DEVICE)
    print('a')
    results_transcriptions, transcription = perform_transcription(transcriber=transcriber, input_file=last_audiofile, lang=source_lan)
    raw_transcripted_text = transcription['text']
    print(3)
    if use_diarization:
        # 4. Diarization. Load diarization model
        diarization = Pipeline.from_pretrained('pyannote/speaker-diarization@2.1', use_auth_token=tk.hf_token)
        # Perform the diarization and store it in a txt file
        # TODO: FIGURE OUT NUM OF SPEAKERS (MIN AND MAX) if not provided
        dz = diarizate_audio_file(diarization, input_file=last_audiofile, nspeakers=num_speakers, store_intermediate_results_folder=temp_folder)
        
        # 5. Speaker recognition. Align timestamps from whisper and diarization.
        results = speaker_recognition(results_transcriptions, dz, unk_strat='closest')
        # Debugging
        paint_results(words=results)

        # form final result
        final_transcription_sentences = build_result_transcription(results)
    else:
        large_single_sentence = rc.Trans_sentence(words=results_transcriptions,
                                speaker=None,
                                sentence_id=0)
        final_transcription_sentences = rc.get_splitted_sentence([large_single_sentence])
        """
        final_transcription_sentences = []
        last_offset = 0
        sent_idx = 0
        for elem in transcription['segments']:
            print(sent_idx)
            num_words = len(elem['text'].split(' '))
            new_offset = num_words+last_offset
            final_transcription_sentences.append(
                rc.Trans_sentence(words=results_transcriptions[last_offset:new_offset],
                                speaker=None,
                                sentence_id=sent_idx))
            sent_idx += 1
            last_offset = new_offset
        """
    print(4)
    # Debugging
    #print(transcripted_text)
    #print('='*50)
    #[print(sentence) for sentence in final_transcription_sentences]

    # 5. Form final results

    # raw text
    output_raw_text = os.path.join(output_folder, 'raw_transcription.txt')
    with open(output_raw_text, 'w') as f:
        f.write(raw_transcripted_text)
    
    transcripted_text = []
    with open(os.path.join(output_folder, 'transcription.txt'), 'w') as f:
        for sentence in final_transcription_sentences:
            transcripted_text.append(f'[{sentence.start_sent}-{sentence.end_sent}] {sentence.spk_text}')
            f.write(sentence.spk_text)
            f.write('\n')
    print(f'Results stored in:{output_folder}/transcription.txt')
    shutil.rmtree(temp_folder)

    output_srt = os.path.join(output_folder, 'transcription.srt')
    sub = generate_srt(sentences=final_transcription_sentences,
                        output_file=output_srt,
                        ann_offset=ap.SPACER_MILLI)

    output_video = os.path.join(output_folder, 'result.mp4')
    burn_subtitles(input_file=original_file, source_type=source_type, input_srt=os.path.join(output_folder, 'transcription.srt'), output_video=output_video)

    # End measure time
    end_t = time.time() - start_t
    print(f'Inference time with unk device: {end_t}')

    return transcripted_text, output_raw_text, output_srt, output_video

if __name__ =='__main__':
    print(pipeline(original_file='input/original.mp4', source_type='audio', model_type='medium', use_diarization=False))