 from transformers import BertModel, BertTokenizer
class IEMOCAPDataset(object):
    def __init__(self, config, data_list):
        self.data_list = data_list
        self.vocabulary_dict = pickle.load(open("E:\data\iemocap\glove300d_w2i.pkl", 'rb'))
        self.audio_length = 3000
        self.feature_name ='fbank'
        self.feature_dim = 40

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        audio_path, asr_text, align_path, label,waveform = self.data_list[index]
        audio_name = os.path.basename(audio_path)
        # ------------- extract the audio features -------------#
        waveform = torch.tensor(waveform)
        _, sample_rate = torchaudio.load(audio_path)
        audio_input = torchaudio.compliance.kaldi.fbank(
                waveform, sample_frequency=sample_rate, num_mel_bins=self.feature_dim,
                frame_length=25, frame_shift=10, use_log_fbank=True)

        # -----------cas1-------------------------------------
        audio_input = audio_input[:self.audio_length, :]
        audio_length = audio_input.size(0)
        delta = torch.tensor(librosa.feature.delta(audio_input))
        delta_seconde = torch.tensor(librosa.feature.delta(audio_input))
        audio_input_n = torch.stack((audio_input, delta, delta), 0)
        audio_input_n2 = torch.cat((audio_input, delta, delta), dim=1)
        audio_input =  audio_input_n2

        # ------------- extract the text contexts -------------#
        text_input_model = tokenizer(asr_text, return_tensors='pt')
        text_input = torch.tensor(tokenizer.encode(asr_text, add_special_tokens=True))
        # Here we use the 0 to represent the padding tokens
        text_length = len(tokenizer.encode(asr_text, add_special_tokens=True))
        #print(text_length)
        #print(text_input_model, text_length)

  
        # ------------- wrap up all the output info the dict format -------------#
        return {'audio_input': audio_input, 'text':asr_text,'text_input': text_input, 'text_input_model': text_input_model,'audio_length': audio_length,
                'text_length': text_length, 'label': label, 'audio_name': audio_name}


def collate(sample_list):

    batch_audio = [x['audio_input'] for x in sample_list]
    batch_asrtext = [x['text'] for x in sample_list]
    batch_text = [x['text_input'] for x in sample_list]
    batch_text2 = [x['text_input_model'] for x in sample_list]
    batch_audio = pad_sequence(batch_audio, batch_first=True)
    batch_text = pad_sequence(batch_text, batch_first=True)
    audio_length = torch.LongTensor([x['audio_length'] for x in sample_list])
    text_length = torch.LongTensor([x['text_length'] for x in sample_list])

    batch_label = torch.tensor([x['label'] for x in sample_list], dtype=torch.long)
    batch_name = [x['audio_name'] for x in sample_list]
    return ((batch_audio, audio_length), (batch_asrtext,batch_text, batch_text2, text_length)), batch_label, batch_name
def tmp_func(x):
    return collate(x)
