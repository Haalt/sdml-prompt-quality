import torch
from torch.utils.data import Dataset


class PromptDataset(Dataset):
    """
    Each item in `samples` is a dict:
        {
          "tokens": LongTensor (T,),      # padded with 0
          "token_mask": FloatTensor (T,),
          "lora_ids": LongTensor (L,),    # padded with 0
          "lora_w": FloatTensor (L,),
          "cfg": float,                   # already normalised
          "n_loras": float,               # already normalised
          "sampler_id": int,              # sampler token ID
          "steps_log": float,             # log-scaled steps [0-1]
          "steps_bucket": int,            # bucketed steps
          "upscaler_id": int,             # upscaler token ID
          "up_has": float,                # whether upscaler is used {0,1}
          "up_steps": float,              # upscaler steps [0-1]
          "denoise": float,               # denoising strength [0-1]
          "target": float                 # quality score 0-1
        }
    All tensors can also be numpy arrays, they will be converted.
    """

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return self.samples[0].shape[0]

    def __getitem__(self, idx):
        item = {
            "tokens": torch.as_tensor(self.samples[0][idx], dtype=torch.long),
            "token_mask": torch.as_tensor(self.samples[1][idx], dtype=torch.float32),
            "lora_ids": torch.as_tensor(self.samples[2][idx], dtype=torch.long),
            "lora_w": torch.as_tensor(self.samples[3][idx], dtype=torch.float32),
            "cfg": torch.as_tensor([self.samples[4][idx]], dtype=torch.float32),
            "n_loras": torch.as_tensor([self.samples[5][idx]], dtype=torch.float32),
            "sampler_id": torch.as_tensor([self.samples[6][idx]], dtype=torch.long),
            "steps_log": torch.as_tensor([self.samples[7][idx]], dtype=torch.float32),
            "steps_bucket": torch.as_tensor([self.samples[8][idx]], dtype=torch.long),
            "upscaler_id": torch.as_tensor([self.samples[9][idx]], dtype=torch.long),
            "up_has": torch.as_tensor([self.samples[10][idx]], dtype=torch.float32),
            "up_steps": torch.as_tensor([self.samples[11][idx]], dtype=torch.float32),
            "denoise": torch.as_tensor([self.samples[12][idx]], dtype=torch.float32),
            "target": torch.as_tensor([self.samples[13][idx]], dtype=torch.float32),
        }
        return item
