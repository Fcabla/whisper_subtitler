from pydub import AudioSegment
import soundfile
import ffmpeg
SPACER_MILLI = 2000
def add_init_spacer(input_file, output_file):
    """ Function that adds and spacer before the audio so pyannote can work better."""
    spacermilli = SPACER_MILLI
    if spacermilli > 0:
        spacer = AudioSegment.silent(duration=spacermilli)
        audio = AudioSegment.from_wav(input_file)
        audio = spacer.append(audio, crossfade=0)
        audio.export(output_file, format='wav')

def file_to_wav(input_file, output_file):
    """ Function to read an audio file (INPUT_FILE) and stores it as a wav file (AUDIO_FILE)."""
    audio_format = input_file[-3:]
    if True:#audio_format != 'wav':
        print(f'switching format from {audio_format} to wav')
        audio = AudioSegment.from_file(input_file, format=audio_format)
        audio.export(output_file, format='wav')

def save_results_from_array(sr, array_audio, filename):
    soundfile.write(file=filename, data=array_audio.squeeze(), samplerate=sr)

def change_sampling_rate(input_file, output_file, new_sr):
    sound = AudioSegment.from_file(input_file,format="wav")
    sound = sound.set_frame_rate(new_sr)
    sound.export(output_file,format="wav")

def change_stereo2mono(input_file, output_file):
    stereo_audio = AudioSegment.from_file(input_file,format="wav")
    mono_audios = stereo_audio.split_to_mono()
    mono_audios[0].export(f"{output_file}_left.wav",format="wav")
    mono_audios[1].export(f"{output_file}_right.wav",format="wav")

def millisec(timeStr):
    spl = timeStr.split(":")
    s = (int)((int(spl[0]) * 60 * 60 + int(spl[1]) * 60 + float(spl[2]) )* 1000)
    return s

def burn_subtitles(input_video, input_srt, output_video):
    video = ffmpeg.input(input_video)
    audio = video.audio
    ffmpeg.concat(video.filter("subtitles", input_srt), audio, v=1, a=1).output(output_video).run()
    #ffmpeg -i subtitles.srt subtitles.ass
    #ffmpeg -i mymovie.mp4 -vf ass=subtitles.ass mysubtitledmovie.mp4