import torch
import torch.nn as nn


class Bart(nn.Module):
    def __init__(self, model_repo, model_name):
        super().__init__()

        bart = torch.hub.load(model_repo, model_name)
        self.bart = bart

    def encode(self, text):
        tokens = self.bart.encode(text)
        return tokens

    def extract_features(self, tokens, return_all_hiddens=False):
        last_layer_features = self.bart.extract_features(tokens, return_all_hiddens)

        return last_layer_features


if __name__ == "__main__":
    bart = Bart(model_repo="pytorch/fairseq", model_name="bart.large")
    text = "This is the test phrase"
    tokens = bart.encode(text)
    features = bart.extract_features(tokens=tokens)
    print("output token : ", tokens.shape)
    print("output features : ", features.shape)
    print("input : ", text)
