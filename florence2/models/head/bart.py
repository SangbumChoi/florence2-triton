import torch.nn as nn

from transformers import BartForConditionalGeneration, BartTokenizer


class Bart(nn.Module):
    def __init__(self, model_name):
        super().__init__()

        self.model = BartForConditionalGeneration.from_pretrained(
            model_name, forced_bos_token_id=0
        )
        self.tokenizer = BartTokenizer.from_pretrained(model_name)

    def encode(self, text):
        tokens = self.tokenizer(text, return_tensors="pt")
        return tokens

    def extract_embedding(self, tokens):
        embedding = self.model.model.shared(tokens)
        return embedding

    def forward(self, inputs_embeds, decoder_input_ids):
        output = self.model(
            inputs_embeds=inputs_embeds, decoder_input_ids=decoder_input_ids
        )
        return output


if __name__ == "__main__":
    bart = Bart(model_name="facebook/bart-large")
    text = "This is the test phrase"
    tokens = bart.encode(text)["input_ids"]
    embedding = bart.extract_embedding(tokens=tokens)
    output = bart.forward(inputs_embeds=embedding, decoder_input_ids=tokens)
    print("input : ", text)
    print("output token : ", tokens.shape)
    print("output embedding : ", embedding.shape)
    print("output : ", output)
