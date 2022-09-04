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
        print(output)
        if self.type == 'bert':
            output = self.classifier(output.last_hidden_state)
            s_output = output.size()
            output = output.view(s_output[0], -1)
        else:
            output = output.logits
        return output


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