import torch.optim
import torch.utils.data
import numpy as np
import torch
import torch.optim
import torch.utils.data
import torch.distributions
from utils.audio.pitch.utils import norm_interp_f0, denorm_f0
from utils.commons.dataset_utils import BaseDataset, collate_1d_or_2d
from utils.commons.indexed_datasets import IndexedDataset
from utils.spec_aug.time_mask import generate_time_mask, generate_alignment_aware_time_mask, generate_inference_mask, generate_continuous_alignment_aware_time_mask


class BaseSpeechDataset(BaseDataset):
    def __init__(self, prefix, shuffle=False, items=None, data_dir=None):
        super().__init__(shuffle)
        from utils.commons.hparams import hparams
        self.data_dir = hparams['directory'] + '/' + hparams['binary_data_dir'] if data_dir is None else data_dir
        # self.data_dir = hparams['binary_data_dir'] if data_dir is None else data_dir
        self.prefix = prefix
        self.hparams = hparams
        self.indexed_ds = None
        if items is not None:
            self.indexed_ds = items
            self.sizes = [1] * len(items)
            self.avail_idxs = list(range(len(self.sizes)))
        else:
            self.sizes = np.load(f'{self.data_dir}/{self.prefix}_lengths.npy')
            if prefix == 'test' and len(hparams['test_ids']) > 0:
                self.avail_idxs = hparams['test_ids']
            else:
                self.avail_idxs = list(range(len(self.sizes)))
            if prefix == 'train' and hparams['min_frames'] > 0:
                self.avail_idxs = [x for x in self.avail_idxs if self.sizes[x] >= hparams['min_frames']]
            self.sizes = [self.sizes[i] for i in self.avail_idxs]

    def _get_item(self, index):
        if hasattr(self, 'avail_idxs') and self.avail_idxs is not None:
            index = self.avail_idxs[index]
        if self.indexed_ds is None:
            self.indexed_ds = IndexedDataset(f'{self.data_dir}/{self.prefix}')
        return self.indexed_ds[index]

    def __getitem__(self, index):
        hparams = self.hparams
        item = self._get_item(index)
        assert len(item['mel']) == self.sizes[index], (len(item['mel']), self.sizes[index])
        max_frames = hparams['max_frames']
        spec = torch.Tensor(item['mel'])[:max_frames]
        max_frames = spec.shape[0] // hparams['frames_multiple'] * hparams['frames_multiple']
        spec = spec[:max_frames]
        ph_token = torch.LongTensor(item['ph_token'][:hparams['max_input_tokens']])
        sample = {
            "id": index,
            "item_name": item['item_name'],
            "text": item['txt'],
            "txt_token": ph_token,
            "mel": spec,
            "mel_nonpadding": spec.abs().sum(-1) > 0,
        }
        if hparams['use_spk_embed']:
            sample["spk_embed"] = torch.Tensor(item['spk_embed'])
        if hparams['use_spk_id']:
            sample["spk_id"] = int(item['spk_id'])
        return sample

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        hparams = self.hparams
        id = torch.LongTensor([s['id'] for s in samples])
        item_names = [s['item_name'] for s in samples]
        text = [s['text'] for s in samples]
        txt_tokens = collate_1d_or_2d([s['txt_token'] for s in samples], 0)
        mels = collate_1d_or_2d([s['mel'] for s in samples], 0.0)
        txt_lengths = torch.LongTensor([s['txt_token'].numel() for s in samples])
        mel_lengths = torch.LongTensor([s['mel'].shape[0] for s in samples])

        batch = {
            'id': id,
            'item_name': item_names,
            'nsamples': len(samples),
            'text': text,
            'txt_tokens': txt_tokens,
            'txt_lengths': txt_lengths,
            'mels': mels,
            'mel_lengths': mel_lengths,
        }

        if hparams['use_spk_embed']:
            spk_embed = torch.stack([s['spk_embed'] for s in samples])
            batch['spk_embed'] = spk_embed
        if hparams['use_spk_id']:
            spk_ids = torch.LongTensor([s['spk_id'] for s in samples])
            batch['spk_ids'] = spk_ids
        return batch


class StutterSpeechDataset(BaseSpeechDataset):
    def __getitem__(self, index):
        sample = super(StutterSpeechDataset, self).__getitem__(index)
        item = self._get_item(index)
        sample['wav_fn'] = item['wav_fn']
        mel = sample['mel']
        T = mel.shape[0]
        sample['mel2ph'] = mel2ph = torch.LongTensor(item['mel2ph'])[:T]
        sample['ph2word'] = ph2word = torch.LongTensor(item['ph2word'])[:T]
        sample['mel2word'] = mel2word = torch.LongTensor(item['mel2word'])[:T]
        max_frames = sample['mel'].shape[0]

        ph_token = sample['txt_token']
        if self.hparams['use_pitch_embed']:
            assert 'f0' in item
            pitch = torch.LongTensor(item.get(self.hparams.get('pitch_key', 'pitch')))[:T]
            f0, uv = norm_interp_f0(item["f0"][:T])
            self.tensor = torch.FloatTensor(uv)
            uv = self.tensor
            f0 = torch.FloatTensor(f0)
            if self.hparams['pitch_type'] == 'ph':
                if "f0_ph" in item:
                    f0 = torch.FloatTensor(item['f0_ph'])
                else:
                    f0 = denorm_f0(f0, None)
                f0_phlevel_sum = torch.zeros_like(ph_token).float().scatter_add(0, mel2ph - 1, f0)
                f0_phlevel_num = torch.zeros_like(ph_token).float().scatter_add(
                    0, mel2ph - 1, torch.ones_like(f0)).clamp_min(1)
                f0_ph = f0_phlevel_sum / f0_phlevel_num
                f0, uv = norm_interp_f0(f0_ph)

                # if "pitch_ph" in item:
                #     pitch = torch.FloatTensor(item['pitch_ph'])
                # else:
                #     pitch = get_phoneme_level_pitch(ph_token, mel2ph, pitch)
                # f0 = torch.gather(f0, 0, mel2ph)
                # uv = torch.gather(uv, 0, mel2ph)

        else:
            f0, uv, pitch = None, None, None
        sample["f0"], sample["uv"], sample["pitch"] = f0, uv, pitch

        if self.hparams['use_energy_embed']:
            assert item['energy'] is not None
            energy = torch.FloatTensor(item.get(self.hparams.get('energy_key', 'energy')))[:T]
            if "energy_ph" in item:
                sample["energy"] = torch.FloatTensor(item['energy_ph'])
            else:
                sample["energy"] = energy

        # Load stutter mask & generate time mask for speech editing
        if 'stutter_mel_mask' in item:
            sample['stutter_mel_mask'] = torch.LongTensor(item['stutter_mel_mask'][:max_frames])
        if self.hparams['infer'] == False:
            mask_ratio = self.hparams['training_mask_ratio']
        else:
            mask_ratio = self.hparams['infer_mask_ratio']

        if self.hparams['infer'] == False:
            if self.hparams.get('mask_type') == 'random':
                time_mel_mask = generate_time_mask(torch.zeros_like(sample['mel']), ratio=mask_ratio)
            elif self.hparams.get('mask_type') == 'alignment_aware':
                # 使用 generate_alignment_aware_time_mask 函数生成对齐感知的掩码
                time_mel_mask = generate_alignment_aware_time_mask(torch.zeros_like(sample['mel']), sample['mel2ph'], ratio=mask_ratio)
            elif self.hparams.get('mask_type') == 'continuous_':
                time_mel_mask = generate_continuous_alignment_aware_time_mask(torch.zeros_like(sample['mel']), sample['mel2ph'], ratio=mask_ratio)
        else:
            # In inference stage we randomly mask the 30% phoneme spans
            time_mel_mask = generate_inference_mask(torch.zeros_like(sample['mel']), sample['mel2ph'], ratio=0.5)
        sample['time_mel_mask'] = time_mel_mask
        return sample

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        batch = super(StutterSpeechDataset, self).collater(samples)
        batch['wav_fn'] = [s['wav_fn'] for s in samples]

        if self.hparams['use_pitch_embed']:
            f0 = collate_1d_or_2d([s['f0'] for s in samples], 0.0)
            pitch = collate_1d_or_2d([s['pitch'] for s in samples])
            uv = collate_1d_or_2d([s['uv'] for s in samples])
        else:
            f0, uv, pitch = None, None, None
        if self.hparams['use_energy_embed']:
            energy = collate_1d_or_2d([s['energy'] for s in samples])
        else:
            energy = None
        mel2ph = collate_1d_or_2d([s['mel2ph'] for s in samples], 0.0)
        ph2word = collate_1d_or_2d([s['ph2word'] for s in samples], 0.0)
        mel2word = collate_1d_or_2d([s['mel2word'] for s in samples], 0.0)

        batch.update({
            'mel2ph': mel2ph,
            'ph2word': ph2word,
            'mel2word': mel2word,
            'pitch': pitch,
            'f0': f0,
            'uv': uv,
            'energy': energy
        })
        if 'stutter_mel_mask' in samples[0]:
            batch['stutter_mel_masks'] = collate_1d_or_2d([s['stutter_mel_mask'] for s in samples], self.hparams.get('stutter_pad_idx', -1))
        batch['time_mel_masks'] = collate_1d_or_2d([s['time_mel_mask'] for s in samples], 0)
        return batch
