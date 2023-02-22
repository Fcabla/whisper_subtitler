import streamlit as st
from pytube import YouTube
import io
import transcriber_sw, transcriber_wx
INPUT_PATH = 'input/'
#@st.cache(suppress_st_warning=True)
# sidebar

# Session
if 'original_file' not in st.session_state:
    st.session_state.original_file = None
if 'source_type' not in st.session_state:
    st.session_state.source_type = None
if 'input_source' not in st.session_state:
    st.session_state.input_source = None
if 'source_lan' not in st.session_state:
    st.session_state.source_lan = None
if 'model_version' not in st.session_state:
    st.session_state.model_version = None
if 'use_diarization' not in st.session_state:
    st.session_state.use_diarization = None
if 'num_speakers' not in st.session_state:
    st.session_state.num_speakers = None
if 'transcripted_text' not in st.session_state:
    st.session_state.transcripted_text = None
if 'output_raw_text' not in st.session_state:
    st.session_state.output_raw_text = None
if 'output_srt' not in st.session_state:
    st.session_state.output_srt = None
if 'output_video' not in st.session_state:
    st.session_state.output_video = None

def load_input(input_source_type, input_source):
    source_type = None
    if input_source is not None:
        if input_source_type == 'Youtube':
            youtubeObject = YouTube(input_source)
            youtubeObject = youtubeObject.streams.get_highest_resolution()
            try:
                st.caption(youtubeObject.title)
                original_file = youtubeObject.download(output_path=INPUT_PATH, filename='original.mp4')
                original_file = f'{INPUT_PATH}original.mp4'
                source_type = 'video'
            except:
                print("An error has occurred")

        elif input_source_type == 'Upload file':
            # Download the file
            original_file = f'{INPUT_PATH}original.{input_source.type[-3:]}'
            g = io.BytesIO(input_source.read())  ## BytesIO Object
            with open(original_file, 'wb') as out:  ## Open temporary file as bytes
                out.write(g.read())  ## Read bytes into file
            
            # Show the file
            if 'video' in input_source.type:
                #st.video(g, format='video/mp4')
                source_type = 'video'
            elif 'audio' in input_source.type:
                #st.audio(g, format='audio/wav')
                source_type = 'audio'
        else:
            print('error')
            exit()
    return original_file, source_type 

with st.sidebar:
    st.title("Input parameters")

    input_source_type = st.radio(label='Input source', options=['Youtube','Upload file'], index=1, horizontal=True)
    if input_source_type == 'Youtube':
        st.session_state.input_source = st.text_input(label='input youtube link')
    elif input_source_type == 'Upload file':
        st.session_state.input_source = st.file_uploader('Upload your file', type=['wav', 'mp3', 'mp4'])
    load_input_source = st.button(label='Load input source')
    st.session_state.source_lan = st.selectbox(label='Select language', options=['unk','en','es'], index=1)
    st.session_state.model_version = st.selectbox(label='Select model version', options=['tiny','base','small','medium', 'large'], index=3)
    st.session_state.use_diarization = st.checkbox(label='Diarization', value=True)
    st.session_state.num_speakers = st.number_input(label='Number of speakers', min_value=0, max_value=5, value=1, help='Input 0 if number of speakers not known', )
    transcribe_button = st.button(label='Start transcription')

#############
# Main page #
#############
st.title("Whisper transcriber & subtitler")
st.caption("Generate transcriptions and subtitles using OpenAI whisper as a base model, stable-ts/whisperx as a timestamp stabilizer using ASR models and pyannote/nemo models in order to identify different speakers.")

# Input Preview
st.header("Input File preview")
if load_input_source:
    st.session_state.original_file, st.session_state.source_type = load_input(input_source_type, st.session_state.input_source)

if st.session_state.original_file and st.session_state.source_type and st.session_state.input_source:
    #g = io.BytesIO(st.session_state.input_source.read())  ## BytesIO Object
    g = open(st.session_state.original_file, 'rb').read() #reading the file
    # Show the file
    if st.session_state.source_type == 'video':
        st.video(g, format='video/mp4')
    elif st.session_state.source_type == 'audio':
        st.audio(g, format='audio/wav')
    st.caption(st.session_state.input_source)

# Transcription results
st.header("Transcription")
if transcribe_button:
    if st.session_state.original_file and st.session_state.source_type:
        print(st.session_state.original_file, st.session_state.source_type)
        result_object = transcriber_wx.pipeline(original_file=st.session_state.original_file, 
                                                source_type=st.session_state.source_type,
                                                source_lan=st.session_state.source_lan, 
                                                model_type=st.session_state.model_version, 
                                                use_diarization=st.session_state.use_diarization,
                                                num_speakers=st.session_state.num_speakers)

        # Collect results
        st.session_state.transcripted_text = result_object['transcripted_text']
        st.session_state.output_raw_text = result_object['output_raw_text']
        st.session_state.output_srt = result_object['output_srt']
        st.session_state.output_video = result_object['output_video']

    else:
        st.error('Remember to input the audio or video and click in the "Load input source"', icon="🚨")

#show results
    
if st.session_state.output_video:
    st.subheader('Result Media')
    video_bytes = open(st.session_state.output_video, 'rb').read() 
    st.video(video_bytes, format='video/mp4') #displaying the video
    st.download_button(label='Download video file', data=video_bytes, file_name='video_result_transcription.mp4', mime='video/mp4')

if st.session_state.output_raw_text:
    st.subheader('Raw transcription')
    raw_text_bytes = open(st.session_state.output_raw_text, 'rb').read()
    st.markdown(raw_text_bytes.decode('utf-8'))
    st.download_button(label='Download raw transcription file', data=raw_text_bytes, file_name='raw_transcription.txt', mime='text/txt')

if st.session_state.transcripted_text:
    st.subheader('Timestamped transcription')
    for elem in st.session_state.transcripted_text:
        st.markdown(elem)
if st.session_state.output_srt:
    srt_bytes = open(st.session_state.output_srt, 'rb').read() 
    st.download_button(label='Download srt file', data=srt_bytes, file_name='transcription.srt', mime='text/plain')