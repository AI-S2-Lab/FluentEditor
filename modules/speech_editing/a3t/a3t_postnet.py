import torch
import torch.nn.functional as F

class Postnet(torch.nn.Module):
    """Postnet module for Spectrogram prediction network.
    This is a module of Postnet in Spectrogram prediction network,
    which described in `Natural TTS Synthesis by
    Conditioning WaveNet on Mel Spectrogram Predictions`_.
    The Postnet predicts refines the predicted
    Mel-filterbank of the decoder,
    which helps to compensate the detail structure of spectrogram.
    .. _`Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions`:
       https://arxiv.org/abs/1712.05884
    """

    def __init__(
        self,
        idim,
        odim,
        n_layers=5,
        n_chans=256,
        n_filts=5,
        dropout_rate=0.0,
        use_batch_norm=True,
    ):
        """Initialize postnet module.
        Args:
            idim (int): Dimension of the inputs.
            odim (int): Dimension of the outputs.
            n_layers (int, optional): The number of layers.
            n_filts (int, optional): The number of filter size.
            n_units (int, optional): The number of filter channels.
            use_batch_norm (bool, optional): Whether to use batch normalization..
            dropout_rate (float, optional): Dropout rate..
        """
        super(Postnet, self).__init__()
        self.postnet = torch.nn.ModuleList()
        for layer in range(n_layers - 1):
            ichans = idim if layer == 0 else n_chans
            ochans = odim if layer == n_layers - 1 else n_chans
            if use_batch_norm:
                self.postnet += [
                    torch.nn.Sequential(
                        torch.nn.Conv1d(
                            ichans,
                            ochans,
                            n_filts,
                            stride=1,
                            padding=(n_filts - 1) // 2,
                            bias=False,
                        ),
                        torch.nn.BatchNorm1d(ochans),
                        torch.nn.Tanh(),
                        torch.nn.Dropout(dropout_rate),
                    )
                ]
            else:
                self.postnet += [
                    torch.nn.Sequential(
                        torch.nn.Conv1d(
                            ichans,
                            ochans,
                            n_filts,
                            stride=1,
                            padding=(n_filts - 1) // 2,
                            bias=False,
                        ),
                        torch.nn.Tanh(),
                        torch.nn.Dropout(dropout_rate),
                    )
                ]
        ichans = n_chans if n_layers != 1 else odim
        if use_batch_norm:
            self.postnet += [
                torch.nn.Sequential(
                    torch.nn.Conv1d(
                        ichans,
                        odim,
                        n_filts,
                        stride=1,
                        padding=(n_filts - 1) // 2,
                        bias=False,
                    ),
                    torch.nn.BatchNorm1d(odim),
                    torch.nn.Dropout(dropout_rate),
                )
            ]
        else:
            self.postnet += [
                torch.nn.Sequential(
                    torch.nn.Conv1d(
                        ichans,
                        odim,
                        n_filts,
                        stride=1,
                        padding=(n_filts - 1) // 2,
                        bias=False,
                    ),
                    torch.nn.Dropout(dropout_rate),
                )
            ]

    def forward(self, xs):
        """Calculate forward propagation.
        Args:
            xs (Tensor): Batch of the sequences of padded input tensors (B, idim, Tmax).
        Returns:
            Tensor: Batch of padded output tensor. (B, odim, Tmax).
        """
        xs = xs.transpose(1,2)
        for i in range(len(self.postnet)):
            xs = self.postnet[i](xs)
        xs = xs.transpose(1,2)
        return xs