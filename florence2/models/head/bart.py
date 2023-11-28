import torch
import torch.nn as nn


class Bart(nn.Module):
    def __init__(self, model_name):
        bart = torch.hub.load("pytorch/fairseq", model_name)
        self.bart = bart

    def encode(self, text):
        tokens = self.bart.encode(text)
        assert tokens.tolist() == [0, 31414, 232, 328, 2]

        return tokens

    def extract_features(self, tokens, return_all_hiddens=False):
        last_layer_features = self.bart.extract_features(tokens, return_all_hiddens)
        assert last_layer_features.size() == torch.Size([1, 5, 1024])

        return last_layer_features
