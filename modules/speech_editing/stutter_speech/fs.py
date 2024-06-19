from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F
from modules.commons.conv import TextConvEncoder, ConvBlocks
from modules.commons.layers import Embedding
from modules.commons.nar_tts_modules import PitchPredictor, DurationPredictor, LengthRegulator
from modules.commons.rel_transformer import RelTransformerEncoder
from modules.commons.rnn import TacotronEncoder, RNNEncoder, DecoderRNN
from modules.commons.transformer import FastSpeechEncoder, FastSpeechDecoder
from modules.commons.wavenet import WN
from modules.tts.commons.align_ops import clip_mel2token_to_multiple, expand_states
from utils.audio.pitch.utils import denorm_f0, f0_to_coarse
from utils.audio.align import mel2token_to_dur

FS_ENCODERS = {
    'fft': lambda hp, dict_size: FastSpeechEncoder(
        dict_size, hp['hidden_size'], hp['enc_layers'], hp['enc_ffn_kernel_size'],
        num_heads=hp['num_heads']),
    'tacotron': lambda hp, dict_size: TacotronEncoder(
        hp['hidden_size'], dict_size, hp['hidden_size'],
        K=hp['encoder_K'], num_highways=4, dropout=hp['dropout']),
    'tacotron2': lambda hp, dict_size: RNNEncoder(dict_size, hp['hidden_size']),
    'conv': lambda hp, dict_size: TextConvEncoder(dict_size, hp['hidden_size'], hp['hidden_size'],
                                                  hp['enc_dilations'], hp['enc_kernel_size'],
                                                  layers_in_block=hp['layers_in_block'],
                                                  norm_type=hp['enc_dec_norm'],
                                                  post_net_kernel=hp.get('enc_post_net_kernel', 3)),
    'rel_fft': lambda hp, dict_size: RelTransformerEncoder(
        dict_size, hp['hidden_size'], hp['hidden_size'],
        hp['ffn_hidden_size'], hp['num_heads'], hp['enc_layers'],
        hp['enc_ffn_kernel_size'], hp['dropout'], prenet=hp['enc_prenet'], pre_ln=hp['enc_pre_ln']),
}

FS_DECODERS = {
    'fft': lambda hp: FastSpeechDecoder(
        hp['hidden_size'], hp['dec_layers'], hp['dec_ffn_kernel_size'], hp['num_heads']),
    'rnn': lambda hp: DecoderRNN(hp['hidden_size'], hp['decoder_rnn_dim'], hp['dropout']),
    'conv': lambda hp: ConvBlocks(hp['hidden_size'], hp['hidden_size'], hp['dec_dilations'],
                                  hp['dec_kernel_size'], layers_in_block=hp['layers_in_block'],
                                  norm_type=hp['enc_dec_norm'], dropout=hp['dropout'],
                                  post_net_kernel=hp.get('dec_post_net_kernel', 3)),
    'wn': lambda hp: WN(hp['hidden_size'], kernel_size=5, dilation_rate=1, n_layers=hp['dec_layers'],
                        is_BTC=True),
}


class FastSpeech(nn.Module):
    def __init__(self, dict_size, hparams, out_dims=None):
        super().__init__()
        self.hparams = deepcopy(hparams)
        self.enc_layers = hparams['enc_layers']
        self.dec_layers = hparams['dec_layers']
        self.hidden_size = hparams['hidden_size']
        self.encoder = FS_ENCODERS[hparams['encoder_type']](hparams, dict_size)
        self.decoder = FS_DECODERS[hparams['decoder_type']](hparams)
        self.out_dims = hparams['audio_num_mel_bins'] if out_dims is None else out_dims
        self.mel_out = nn.Linear(self.hidden_size, self.out_dims, bias=True)
        if hparams['use_spk_id']:
            self.spk_id_proj = Embedding(hparams['num_spk'], self.hidden_size)
        if hparams['use_spk_embed']:
            self.spk_embed_proj = nn.Linear(256, self.hidden_size, bias=True)
        predictor_hidden = hparams['predictor_hidden'] if hparams['predictor_hidden'] > 0 else self.hidden_size
        self.dur_embed = Embedding(2000, self.hidden_size, 0)
        self.dur_predictor = DurationPredictor(
            self.hidden_size,
            n_chans=predictor_hidden,
            n_layers=hparams['dur_predictor_layers'],
            dropout_rate=hparams['predictor_dropout'],
            kernel_size=hparams['dur_predictor_kernel'])
        self.length_regulator = LengthRegulator()
        if hparams['use_pitch_embed']:
            self.pitch_embed = Embedding(300, self.hidden_size, 0)
            self.pitch_predictor = PitchPredictor(
                self.hidden_size, n_chans=predictor_hidden,
                n_layers=5, dropout_rate=0.2, odim=2,
                kernel_size=hparams['predictor_kernel'])
        if hparams['dec_inp_add_noise']:
            self.z_channels = hparams['z_channels']
            self.dec_inp_noise_proj = nn.Linear(self.hidden_size + self.z_channels, self.hidden_size)
        del self.mel_out
        del self.decoder

    def forward(self, txt_tokens, time_mel_masks, mel2ph, spk_embed,
                f0, uv, spk_id=None, skip_decoder=True, infer=False, use_pred_mel2ph=False, use_pred_pitch=False, **kwargs):
        ret = {}
        encoder_out = self.encoder(txt_tokens)  # [B, T, C]
        src_nonpadding = (txt_tokens > 0).float()[:, :, None]
        style_embed = self.forward_style_embed(spk_embed, spk_id)

        # add dur
        dur_inp = (encoder_out + style_embed) * src_nonpadding
        mel2ph = self.forward_dur(dur_inp, time_mel_masks, mel2ph, txt_tokens, ret, use_pred_mel2ph=use_pred_mel2ph)
        tgt_nonpadding = (mel2ph > 0).float()[:, :, None]
        decoder_inp = expand_states(encoder_out, mel2ph)

        # add pitch embed
        if self.hparams['use_pitch_embed']:
            pitch_inp = (decoder_inp + style_embed) * tgt_nonpadding
            decoder_inp = decoder_inp + self.forward_pitch(pitch_inp, time_mel_masks, f0, uv, mel2ph, ret, use_pred_pitch=use_pred_pitch)

        # decoder input
        ret['decoder_inp'] = decoder_inp = (decoder_inp + style_embed) * tgt_nonpadding
        if skip_decoder:
            return ret
        if self.hparams['dec_inp_add_noise']:
            B, T, _ = decoder_inp.shape
            z = kwargs.get('adv_z', torch.randn([B, T, self.z_channels])).to(decoder_inp.device)
            ret['adv_z'] = z
            decoder_inp = torch.cat([decoder_inp, z], -1)
            decoder_inp = self.dec_inp_noise_proj(decoder_inp) * tgt_nonpadding
        ret['mel_out'] = self.forward_decoder(decoder_inp, tgt_nonpadding, ret, infer=infer, **kwargs)
        return ret

    def forward_style_embed(self, spk_embed=None, spk_id=None):
        # add spk embed
        style_embed = 0
        if self.hparams['use_spk_embed']:
            style_embed = style_embed + self.spk_embed_proj(spk_embed)[:, None, :]
        if self.hparams['use_spk_id']:
            style_embed = style_embed + self.spk_id_proj(spk_id)[:, None, :]
        return style_embed

    def forward_dur(self, dur_input, time_mel_masks, mel2ph, txt_tokens, ret, masked_dur=None, use_pred_mel2ph=False):
        """

        :param dur_input: [B, T_txt, H]
        :param dur_input: [B, T_mel, 1]
        :param mel2ph: [B, T_mel]
        :param txt_tokens: [B, T_txt]
        :param ret:
        :return:
        """
        # Add masked gt dur to dur_input
        if masked_dur is None:
            B, T = txt_tokens.shape
            nonpadding = (txt_tokens != 0).float()
            masked_dur_gt = mel2token_to_dur(mel2ph*(1-time_mel_masks).squeeze(-1).long(), T) * nonpadding
            dur_input = dur_input + self.dur_embed(masked_dur_gt.long())
        else:
            dur_input = dur_input + self.dur_embed(masked_dur.long())

        # Forward duration
        src_padding = txt_tokens == 0
        if self.hparams['predictor_grad'] != 1:
            dur_input = dur_input.detach() + self.hparams['predictor_grad'] * (dur_input - dur_input.detach())
        dur = self.dur_predictor(dur_input, src_padding)
        ret['dur'] = dur
        if use_pred_mel2ph:
            mel2ph = self.length_regulator(dur, src_padding).detach()
        ret['mel2ph'] = mel2ph = clip_mel2token_to_multiple(mel2ph, self.hparams['frames_multiple'])
        return mel2ph

    def forward_pitch(self, decoder_inp, time_mel_masks, f0, uv, mel2ph, ret, use_pred_pitch):
        pitch_pred_inp = decoder_inp
        pitch_padding = mel2ph == 0
        use_uv = self.hparams['pitch_type'] == 'frame' and self.hparams['use_uv']

        # Add masked gt pitch to pitch_pred_inp
        time_mel_masks = time_mel_masks.squeeze(-1)
        masked_f0 = f0*(1-time_mel_masks)
        masked_uv = uv*(1-time_mel_masks)
        masked_gt_f0_denorm = denorm_f0(masked_f0, masked_uv if use_uv else None, pitch_padding=pitch_padding)
        masked_gt_pitch = f0_to_coarse(masked_gt_f0_denorm)  # start from 0 [B, T_mel]
        pitch_pred_inp = pitch_pred_inp + self.pitch_embed(masked_gt_pitch.long())

        # Forward pitch
        if self.hparams['predictor_grad'] != 1:
            pitch_pred_inp = pitch_pred_inp.detach() + \
                             self.hparams['predictor_grad'] * (pitch_pred_inp - pitch_pred_inp.detach())
        ret['pitch_pred'] = pitch_pred = self.pitch_predictor(pitch_pred_inp)
        if use_pred_pitch:
            pitch_padding = None
            pred_f0 = pitch_pred[:, :, 0]
            if use_uv:
                pred_uv = pitch_pred[:, :, 1] > 0
            res_f0 = f0 * (1-time_mel_masks) + pred_f0 * time_mel_masks
            res_uv = uv * (1-time_mel_masks) + pred_uv * time_mel_masks
        else:
            res_f0 = f0
            res_uv = uv

        f0_denorm = denorm_f0(res_f0, res_uv if use_uv else None, pitch_padding=pitch_padding)
        pitch = f0_to_coarse(f0_denorm)  # start from 0 [B, T_mel]
        ret['f0_denorm'] = f0_denorm
        ret['f0_denorm_pred'] = denorm_f0(
            pitch_pred[:, :, 0], (pitch_pred[:, :, 1] > 0) if use_uv else None,
            pitch_padding=pitch_padding)
        pitch_embed = self.pitch_embed(pitch)
        return pitch_embed

    def forward_decoder(self, decoder_inp, tgt_nonpadding, ret, infer, **kwargs):
        x = decoder_inp  # [B, T, H]
        x = self.decoder(x)
        x = self.mel_out(x)
        return x * tgt_nonpadding
