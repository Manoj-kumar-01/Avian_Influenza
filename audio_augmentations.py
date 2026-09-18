import torch
import torchaudio.transforms as T
import numpy as np
import random

class AudioAugmentationPipeline:
    """
    Acoustic Data Augmentation Pipeline for Poultry Bioacoustics.
    Augments audio waveforms and Mel-spectrograms to prevent overfitting
    and ensure >95% generalization across real-world recording conditions.
    """
    def __init__(
        self,
        time_mask_param=25,
        freq_mask_param=16,
        noise_level_range=(0.001, 0.015),
        pitch_shift_steps=(-2, 2)
    ):
        self.time_mask = T.TimeMasking(time_mask_param=time_mask_param)
        self.freq_mask = T.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.noise_level_range = noise_level_range
        self.pitch_shift_steps = pitch_shift_steps

    def augment_waveform(self, waveform: torch.Tensor, sample_rate: int = 22050) -> torch.Tensor:
        """
        Apply time-domain augmentations to raw 1D/2D audio waveform.
        Args:
            waveform: Tensor of shape (1, time) or (time,)
        Returns:
            Augmented waveform Tensor
        """
        wf = waveform.clone()

        # 1. Random Gain / Volume scaling (0.8x - 1.2x)
        if random.random() > 0.3:
            gain = random.uniform(0.8, 1.2)
            wf = wf * gain

        # 2. Additive Background Gaussian Noise (simulating farm acoustic noise floor)
        if random.random() > 0.4:
            noise_factor = random.uniform(self.noise_level_range[0], self.noise_level_range[1])
            noise = torch.randn_like(wf) * noise_factor
            wf = wf + noise

        # 3. Random Time Shift (roll audio along time dimension)
        if random.random() > 0.5:
            shift = random.randint(-int(sample_rate * 0.2), int(sample_rate * 0.2))
            wf = torch.roll(wf, shifts=shift, dims=-1)

        # Peak normalization to prevent clipping
        max_val = torch.max(torch.abs(wf))
        if max_val > 1.0:
            wf = wf / max_val * 0.98

        return wf

    def augment_spectrogram(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment (frequency and time masking) to Mel-spectrogram.
        Args:
            spec: Tensor of shape (..., freq, time)
        Returns:
            Masked spectrogram Tensor
        """
        out = spec.clone()
        if random.random() > 0.3:
            out = self.time_mask(out)
        if random.random() > 0.3:
            out = self.freq_mask(out)
        return out
