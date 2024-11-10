import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch


def make_item_map(train_d_path, val_d_path=None, sep=',', item_key='ItemId'):   #ItemId
    train_df = pd.read_csv(train_d_path, sep=sep, dtype={item_key: int})
    item_ids = train_df[item_key]
    if val_d_path is not None:
        val_df = pd.read_csv(val_d_path, sep=sep, dtype={item_key: int})
        val_item_ids = val_df[item_key]
        item_ids = pd.concat([item_ids, val_item_ids], axis=0)
    item_ids = item_ids.unique()  # type is numpy.ndarray
    item2idx = pd.Series(data=np.arange(len(item_ids)), index=item_ids)
    # Build itemmap is a DataFrame that have 2 columns (self.item_key, 'item_idx')
    itemmap = pd.DataFrame({item_key: item_ids, 'item_idx': item2idx[item_ids].values})
    return itemmap

class SessionDataset(Dataset):
    def __init__(self, path, item_map, sep=',', session_key='SessionId', item_key='ItemId', time_key='Time'):
        # Read csv
        self.df = pd.read_csv(path, sep=sep, dtype={session_key: int, item_key: int, time_key: float})
        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key

        self.itemmap = item_map

        # 增加一列，内容为相应的item的index
        self.add_item_indices()

        # 按SessionID和Time对dataframe进行排序
        #   Sort the df by time, and then by session ID. That is, df is sorted by session ID and
        #   clicks within a session are next to each other, where the clicks within a session are time-ordered.
        self.df.sort_values([session_key, time_key], inplace=True)

        # 各session起始位置
        self.click_offsets = self.get_click_offset()

    def __getitem__(self, index):
        if index >=0:
            start = self.click_offsets[index]
            end = self.click_offsets[index + 1]
        else:
            start = self.click_offsets[index - 1]
            end = self.click_offsets[index]
        return self.df['item_idx'][start: end].to_numpy()

    def __len__(self):
        return len(self.click_offsets) - 1

    def add_item_indices(self):
        """
        向dataframe中添加一列，即item的index列
        """
        self.df = pd.merge(self.df, self.itemmap, on=self.item_key, how='inner')

    def get_click_offset(self):
        """
        返回每个session的第一个项目的位置
        self.df[self.session_key] return a set of session_key
        self.df[self.session_key].nunique() return the size of session_key set (int)
        self.df.groupby(self.session_key).size() return the size of each session_id
        self.df.groupby(self.session_key).size().cumsum() retunn cumulative sum
        """
        offsets = np.zeros(self.df[self.session_key].nunique() + 1, dtype=np.int32)
        offsets[1:] = self.df.groupby(self.session_key).size().cumsum()
        return offsets


class SessionLoader(DataLoader):
    def __init__(self, ds, padd_idx, batch_size, shuffle):
        super().__init__(ds, batch_size, shuffle, collate_fn=self._collate)
        self.padd_idx = padd_idx

    def _collate(self, batch):  #将序列填充到批次中的最大序列长度。
        '''
        batch_x: 这是一个包含了多个输入序列的张量。每个输入序列对应一个会话（或批次中的一个样本）。在这个张量中，所有的输入序列都已经通过填充（padding）对齐到批次中最长的序列的长度，以确保它们具有相同的长度。填充通常使用一个特定的填充索引值（padd_idx）来完成。
        batch_y: 这是一个包含了多个目标序列的张量，它们对应于与 batch_x 中的输入序列相对应的目标序列。目标序列通常是输入序列向左偏移一个时间步，因为在许多序列预测任务中，目标是预测下一个时间步的内容。与 batch_x 一样，所有目标序列也已经填充到相同的长度。
        '''
        max_len = max(len(d) for d in batch)
        batch_x = np.array([self.padd(d, max_len) for d in batch])
        batch_y = np.array([self.padd(d[1:], max_len-1) for d in batch])
        return torch.LongTensor(batch_x), torch.LongTensor(batch_y)

    def padd(self, data, max_len):  #用于填充单独的序列以使其具有相同的长度。
        return np.pad(data, [0, max_len-len(data)],'constant', constant_values=self.padd_idx)
    

