from torch.utils.data import Dataset

class UnlearningDataset(Dataset):
    def __init__(self, forget_data, retain_data):
        assert len(forget_data) == len(retain_data)
        self.forget = forget_data
        self.retain = retain_data

    def __len__(self):
        return len(self.forget)

    def __getitem__(self, idx):
        return {
            "forget": self.forget[idx],
            "retain": self.retain[idx],
        }