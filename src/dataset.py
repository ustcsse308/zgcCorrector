from torch.utils.data import Dataset
import tqdm
import torch
import random

# 23个声母，24个韵母
initials = 'b p m f d t n l ɡ k h j q x zh ch sh r z c s y w'.split(' ')
initial2id = {initial:i+1 for i, initial in enumerate(initials)}

finals = 'a o e i u v ai ei ui ao ou iu ie ue er an en in un vn ang eng ing ong'.split(' ')
final2id = {final:i for i, final in enumerate(finals)}

from pypinyin import lazy_pinyin, Style
print(lazy_pinyin('绿色，',style = Style.TONE3))




def getPictures(char, savePath, fontType, fontPath):
    from PIL import Image, ImageFont, ImageDraw
    image = Image.new('RGB', (250, 250), (255,255,255)) 
    iwidth, iheight = image.size
    font = ImageFont.truetype(fontPath+'/'+fontType, 60)
    draw = ImageDraw.Draw(image)

    fwidth, fheight = draw.textsize(char, font)

    fontx = (iwidth - fwidth - font.getoffset(char)[0]) / 2
    fonty = (iheight - fheight - font.getoffset(char)[1]) / 2

    draw.text((fontx, fonty), char, 'black', font)
    image.save(savePath+char+'.jpg') 

class MultiModalDataset(Dataset):
    def __init__(self, path, if_training=False, maxlen=128, glyphTokenizer=None, semanticTokenizer=None, speechTokenizer=None, detectorTokenizer=None):
        self.maxlen = maxlen
        self.glyphTokenizer = glyphTokenizer
        self.semanticTokenizer = semanticTokenizer
        self.speechTokenizer = speechTokenizer
        self.detectorTokenizer = detectorTokenizer
        self.corpus = []
        with open(path, 'r') as f:
            for line in f.readlines():
                # src tar detect_label
                line.strip().split('\t')
                self.append(line)
        self.if_training = if_training
    
    def __len__(self):
        return len(self.corpus)
    
    def __getitem__(self, idx):
        text_src, text_tar, detect_label = self.corpus[idx]
        len_src = len(text_src)
        
        ret = {}
        
        # Glyph Tokenizer
        glyphInput = self.getGlyphData(text_src)
        ret['glyph_input_ids'] = glyphInput['input_ids']
        ret['glyph_token_type_ids'] = glyphInput['token_type_ids']
        ret['glyph_attention_mask'] = glyphInput['attention_mask']
        
        # Semantic Tokenizer
        semanticInput = self.getSemanticData(text_src)
        ret['semantic_input_ids'] = semanticInput['input_ids']
        ret['semantic_token_type_ids'] = semanticInput['token_type_ids']
        ret['semantic_attention_mask'] = semanticInput['attention_mask']
        
        # Speech Tokenizer
        # TODO(@马承成): 完成时间(2022/09/10)
        
        
        # Detector Tokenizer
        detectorInput = self.getDetectorData(text_src)
        ret['detector_input_ids'] = detectorInput['input_ids']
        ret['detector_token_type_ids'] = detectorInput['token_type_ids']
        ret['detector_attention_mask'] = detectorInput['attention_mask']
        
        
        # label
        # text_tar and detect_label
        label = self.getLabelData(text_tar, detect_label)
        ret['detect_label'] = label['detect_label']
        ret['tar_label'] = label['tar_label']

        return ret
    
    def getLabel(self, text_tar, detect_label):
        tar = self.semanticTokenizer(
            text_tar,
            add_special_tokens=True,
            return_tensors="pt",
            return_token_type_ids=True,
            return_attention_mask=True,
            truncation=True,
            padding=True,
            max_length=self.maxlen
        )
        ret = {}
        ret['tar_label'] = tar['input_ids']
        detect_label = [int(label) for label in detect_label]
        ret['detect_label'] = torch.tensor(detect_label, dtype=torch.long)
        return ret

    def getSpeechData(self, text_src):
        # TODO(@马承成)：完成时间(2022/09/10)
        pass
    
    def getGlyphData(self, text_src):
        return self.glyphTokenizer(
            text_src,
            add_special_tokens=True,
            return_tensors="pt",
            return_token_type_ids=True,
            return_attention_mask=True,
            truncation=True,
            padding=True,
            max_length=self.maxlen
        )
    
    def getSemanticData(self, text_src):
        return self.semanticTokenizer(
            text_src,
            add_special_tokens=True,
            return_tensors="pt",
            return_token_type_ids=True,
            return_attention_mask=True,
            truncation=True,
            padding=True,
            max_length=self.maxlen
        )

    def getDetectorData(self, text_src):
        return self.detectorTokenizer(
            text_src,
            add_special_tokens=True,
            return_tensors="pt",
            return_token_type_ids=True,
            return_attention_mask=True,
            truncation=True,
            padding=True,
            max_length=self.maxlen
        )
        
if __name__ == '__main__':
    pass