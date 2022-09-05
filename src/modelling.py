from torch import nn
import torch
from transformers import BertTokenizer, AutoModel
from transformers import ElectraForPreTraining, ElectraTokenizerFast

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

class DetectModel(nn.Module):
    def __init__(self, DetectModelPath, type='electra'):
        super(DetectModel, self).__init__()
        if type == 'bert':
            self.bert = AutoModel.from_pretrained(DetectModelPath)
        else:
            self.bert = ElectraForPreTraining.from_pretrained(DetectModelPath)
        self.type = type
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, **kwarg):
        output = self.bert(input_ids, token_type_ids, attention_mask)
        if self.type == 'bert':
            output = self.classifier(output.last_hidden_state)
            s_output = output.size()
            output = output.view(s_output[0], -1)
        else:
            output = output.logits
        # output:[batch, seq_len]
        return output

class SemanticModel(nn.Module):
    def __init__(self, SemanticModelPath, type='electra'):
        super(SemanticModel, self).__init__()
        if type == 'bert':
            self.bert = AutoModel.from_pretrained(SemanticModel)
        else:
            self.bert = ElectraForPreTraining.from_pretrained(SemanticModel)
        self.type = type

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, **kwarg):
        output = self.bert(input_ids, token_type_ids, attention_mask)
        if self.type == 'bert':
            output = output.last_hidden_state
        else:
            output = output.logits
        # output:[batch, seq_len, hid]
        return output

# TODO(@马承成): 完成时间2022/09/10
class SpeechModel(nn.Module):
    def __init__(self, SpeechModelPath, type='bert'):
        super(SpeechModel, self).__init__()
        pass
    
    
# TODO(@马承成): 完成时间2022/09/10
class GlyphModel(nn.Module):
    def __init__(self, GlyphModelPath, type='bert'):
        super(GlyphModel, self).__init__()
        pass
    
    

class Corrector(nn.Module):
    def __init__(self, SemanticModelPath, SpeechModelPath, GlyphModelPath, DetectModelPath, typeSemantic='bert', typeSpeech='', typeGlyph='bert', typeDetect='electra'):
        self.GlyphModel = GlyphModel(GlyphModelPath, typeGlyph)
        self.DetectModel = DetectModel(DetectModelPath, typeDetect)
        self.SpeechModel = SpeechModel(SpeechModelPath, typeSpeech)
        self.SemanticModel = SemanticModel(SemanticModelPath, typeSemantic)
    
    def getKL4AttentionMatrix(self, l1, l2, l3):
        # TODO(@zzzgc)
        pass

    def forward(self, x):
        logits4Semantic = self.GlyphModel(input_ids=x['semantic_input_ids'], token_type_ids=x['semantic_input_ids'], attention_mask=x['semantic_input_ids'])
        logits4Glyph = self.GlyphModel(input_ids=x['glyph_input_ids'], token_type_ids=x['glyph_token_type_ids'], attention_mask=x['glyph_attention_mask'])
        # TODO(@马承成)
        logits4Speech = self.SpeechModel(input_ids=x['semantic_input_ids'], token_type_ids=x['semantic_input_ids'], attention_mask=x['semantic_input_ids'])
        logists4Detect = self.DetectModel(input_ids=x['detector_input_ids'], token_type_ids=x['detector_token_type_ids'], attention_mask=x['detector_attention_mask'])
        KLoss = self.getKL4AttentionMatrix(logits4Semantic, logits4Glyph, logits4Speech)
        # TODO(@zzzgc)
        
    
def test_detect():
    from transformers import ElectraForPreTraining, ElectraTokenizerFast
    import torch

    # discriminator = DetectModel("hfl/chinese-electra-180g-large-discriminator")
    discriminator = DetectModel('bert-base-chinese', 'bert')
    # discriminator = ElectraForPreTraining.from_pretrained("hfl/chinese-electra-180g-large-discriminator")
    tokenizer = ElectraTokenizerFast.from_pretrained("hfl/chinese-electra-180g-large-discriminator")

    sentence = "The quick brown fox jumps over the lazy dog"
    fake_sentence = [["我爱背景天安门","我爱背景天安门我爱背景天安门"],["我爱背景天安门","我爱背景天安门我爱背景天安门"]]

    fake_tokens = tokenizer(fake_sentence,return_tensors="pt")
    # fake_inputs = tokenizer.encode(fake_sentence, return_tensors="pt")
    
    # print(fake_tokens)
    
    discriminator_outputs = discriminator(fake_tokens.input_ids, fake_tokens.token_type_ids, fake_tokens.attention_mask)
    print(discriminator_outputs)
    predictions = torch.round((torch.sign(discriminator_outputs[0]) + 1) / 2)

    print(predictions)
    # [print("%7s" % token, end="") for token in fake_tokens]
    # print()
    # [print("%7s" % int(prediction), end="") for prediction in predictions.squeeze().tolist()]
    # print()

if __name__ == "__main__":
    test_detect()