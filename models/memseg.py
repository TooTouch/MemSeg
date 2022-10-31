import torch.nn as nn
from .decoder import Decoder
from .msff import MSFF

class MemSeg(nn.Module):
    def __init__(self, memory_bank, feature_extractor):
        super(MemSeg, self).__init__()

        self.memory_bank = memory_bank
        self.feature_extractor = feature_extractor
        self.msff = MSFF()
        self.decoder = Decoder()

    def forward(self, inputs):
        # extract features
        features = self.feature_extractor(inputs)
        f_in = features[0]
        f_out = features[-1]
        f_ii = features[1:-1]

        # extract concatenated information(CI)
        concat_features = self.memory_bank.select(features = f_ii)

        # Multi-scale Feature Fusion(MSFF) Module
        msff_outputs = self.msff(features = concat_features)

        # decoder
        predicted_mask = self.decoder(
            encoder_output  = f_out,
            concat_features = [f_in] + msff_outputs
        )

        return predicted_mask
