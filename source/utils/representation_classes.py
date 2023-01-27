from nltk import tokenize
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class Trans_word:
    def __init__(self, token, start, id_order, end=None):
        self.token = token
        self.start = round(start,4)
        if end:
            self.end = round(end,4)
        else: 
            self.end = end
        self.id_order = id_order
        self.speaker = 'UNK'

    def set_speaker(self, speaker):
        self.speaker = speaker
    
    def todict(self):
        return {'token':self.token,'start':self.start,'end':self.end,'id_order':self.id_order,'speaker':self.speaker}
        
    def __repr__(self):
        return f"{self.token}, ts:{self.start}-{self.end}, id:{self.id_order}, sk:{self.speaker}"

    def __str__(self):
        return f"{self.token}, ts:{self.start}-{self.end}, id:{self.id_order}, sk:{self.speaker}"

class Trans_sentence:
    def __init__(self, words, speaker, sentence_id):
        self.words = words
        self.speaker = speaker
        self.sentence_id = sentence_id
        self.text = f"{' '.join([word.token for word in self.words])}"
        if speaker:
            self.spk_text = f"{self.speaker}: {self.text}"
        else:
            self.spk_text = self.text
        self.start_sent = self.words[0].start
        if self.words[-1].end:
            self.end_sent = self.words[-1].end
        else:
            # maybe sum threshold like 0.2s
            self.end_sent = self.words[-1].start

    def get_sentence_nospeaker(self):
        return ' '.join([word.token for word in self.words])
    
    def __repr__(self):
        return self.spk_text

    def __str__(self):
        return self.spk_text

def get_splitted_sentence(transcribed_sentences):
    fixed_transcribed_sentences = []
    new_sent_idx = 0
    for transcribed_sentence in transcribed_sentences:
        splitted_sents = tokenize.sent_tokenize(transcribed_sentence.text)
        sent_words = transcribed_sentence.words
        speaker = transcribed_sentence.speaker

        last_offset = 0
        for sent in splitted_sents:
            num_words = len(sent.split(' '))
            new_offset = num_words+last_offset
            fixed_transcribed_sentences.append(
                Trans_sentence(words=sent_words[last_offset:new_offset],
                                speaker=speaker,
                                sentence_id=new_sent_idx))
            new_sent_idx += 1
            last_offset = new_offset
    return fixed_transcribed_sentences

def smooth_starts(transcribed_sentences):
    for sent_idx in range(1,len(transcribed_sentences)):
        previous_end_ts = transcribed_sentences[sent_idx-1].end_sent
        current_start_ts = transcribed_sentences[sent_idx].start_sent

        delta = round(current_start_ts-previous_end_ts, 3)
        transcribed_sentences[sent_idx-1].end_sent += round(delta/2,3)
        transcribed_sentences[sent_idx].start_sent -= round(delta/2,3)
    return transcribed_sentences