# TODO stack S=16 layers

from torch import nn
from preprocessing.process_dataframe import sequence_encoder

class LensDesignCreator(nn.Module):
    def __init__(self, lens_type):
        super().__init__()
        code_lenstype = sequence_encoder(lens_type)
        numsurf = len(str(code_lenstype))
        numglass = sum(map(int, str(code_lenstype)))
        numin = 2 + 2*numsurf
        numout = 2*numglass + 2*numsurf-1

        self.layers = nn.Sequential(
                                nn.Linear(numin, 32), # nn.Linear(入力層の数, 出力層の数)
                                nn.SELU(inplace=True), # inplace=Trueを指定することでメモリを節約できる(推奨)
                                nn.Linear(32, 32),
                                nn.SELU(inplace=True),
                                nn.Linear(32, 32),
                                nn.SELU(inplace=True),
                                nn.Linear(32, 32),
                                nn.SELU(inplace=True),
                                nn.Linear(32, 32),
                                nn.SELU(inplace=True),
                                nn.Linear(32, 32),
                                nn.SELU(inplace=True),
                                nn.Linear(32, 32),
                                nn.SELU(inplace=True),
                                nn.Linear(32, numout)
                            )
    # 順伝搬の計算の定義
    def forward(self, x):
      return self.layers(x)