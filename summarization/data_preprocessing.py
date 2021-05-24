import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class Dataset(Dataset):
    """(본문, 요약문) 구조를 가진 데이터 셋 생성"""

    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        topic = self.df.iloc[idx, 0]
        ans = self.df.iloc[idx, 1]
        return topic, ans


def data_preprocessing(text):
    """국립 국어원 요약 말 뭉치 데이터 전처리"""

    text = pd.DataFrame(text)
    train_data, valid_data, test_data = np.split(text.sample(frac=1,
                                                             random_state=200),
                                                 [int(.8 * len(text)),
                                                  int(.9 * len(text))])

    train_dataset = Dataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,
                              num_workers=2)

    valid_dataset = Dataset(valid_data)
    valid_loader = DataLoader(valid_dataset, batch_size=5, shuffle=True,
                              num_workers=2, drop_last=True)

    test_dataset = Dataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True,
                             num_workers=2)

    return train_loader, valid_loader, test_loader
