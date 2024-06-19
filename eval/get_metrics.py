from glob import glob
import pandas as pd
from eval.mcd import cal_mcd_with_wave_batch
from eval.stoi import cal_stoi_with_waves_batch
from eval.pesq_metric import cal_pesq_with_waves_batch


if __name__ == '__main__':

    
    # VCTK
    # wavs_dir = 'checkpoints/campnet_vctk/generated_2000000_/wavs/*'
    # wavs_dir = 'checkpoints/campnet_vctk_ali_0.8/generated_1000000_/wavs/*'
    # wavs_dir = 'checkpoints/spec_denoiser_vctk_wo_pitch/generated_300000_/wavs/*'
    # wavs_dir = 'checkpoints/spec_denoiser_vctk_normal/generated_300000_/wavs/*'
    # wavs_dir = 'checkpoints/spec_denoiser_vctk_dur_pitch_masked_0.8/generated_300000_/wavs/*'
    # wavs_dir = 'checkpoints/editspeech_orig_0.3_vctk/generated_100000_/wavs/*'
    # wavs_dir = 'checkpoints/a3t_vctk_0.8/generated_800000_/wavs/*'
    wavs_dir = '/checkpoints/fluenteditor/generated_2000000_/wavs/*'
    
    mcd_values = cal_mcd_with_wave_batch(wavs_dir)
    stoi_values = cal_stoi_with_waves_batch(wavs_dir)
    pesq_values = cal_pesq_with_waves_batch(wavs_dir)

    print(f"{wavs_dir}")
    print(f"MCD = {mcd_values}; STOI = {stoi_values}; PESQ = {pesq_values}.")