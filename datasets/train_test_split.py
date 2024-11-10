import pandas as pd
import numpy as np
import random

# 训练session比例
train_p = 0.8
'''
# ---- 数据 30music
d_name = '30music'
d_file = '30music-200ks.csv'
sep='\t'
session_key='SessionId'
item_key='ItemId'
time_key='Time'

# ---- 数据 aotm
d_name = 'aotm'
d_file = 'playlists-aotm.csv'
sep='\t'
session_key='SessionId'
item_key='ItemId'
time_key='Time'

# ---- 数据 diginetica
d_name = 'diginetica'
d_file = 'train-item-views.csv'
sep=';'
session_key='sessionId'
item_key='itemId'
time_key='timeframe'

# ---- 数据 nowplaying
d_name = 'nowplaying'
d_file = 'nowplaying.csv'
sep='\t'
session_key='SessionId'
item_key='ItemId'
time_key='Time'
'''

# ---- 数据 yoochoose(小)
d_name = 'yoochoosemini'
d_file = 'recSys15TrainOnlymini.txt'
sep=','
session_key='SessionID'
item_key='ItemID'
time_key='Time'

# ---- 数据 yoochoose(大)
d_name = 'yoochoose'
d_file = 'recSys15TrainOnly.txt'
sep=','
session_key='SessionID'
item_key='ItemID'
time_key='Time'


d_path = f'./datasets/{d_name}/raw/{d_file}'

def get_offset(df, session_key):
    offsets = np.zeros(df[session_key].nunique() + 1, dtype=np.int32)
    offsets[1:] = df.groupby(session_key).size().cumsum()
    return offsets

df = pd.read_csv(d_path, sep=sep, dtype={session_key: int, item_key: int, time_key: float})
print(df.columns[0])   #补充
# 创建item map
item_ids = df[item_key]
item_ids = item_ids.unique()
item2idx = pd.Series(data=np.arange(len(item_ids)), index=item_ids)
itemmap = pd.DataFrame({item_key: item_ids, 'item_idx': item2idx[item_ids].values})

itemmap.rename(columns={item_key: 'ItemId'}, inplace=True)
itemmap.to_csv(f'./datasets/{d_name}/itemmap.csv', index=False)

# 排序并构建每个session的offsets
df.sort_values([session_key, time_key], inplace=True)
offsets = get_offset(df, session_key)
offsets = np.stack([offsets[:-1], offsets[1:]]).transpose()

# 训练、测试的idx划分
idx = list(range(len(offsets)))
random.shuffle(idx)
train_num = int(len(idx) * train_p)
train_idx = idx[:train_num]
test_idx = idx[train_num:]

# 训练、测试的dataframe
train_df = pd.concat([df.iloc[offsets[idx][0]:offsets[idx][1]] for idx in train_idx])
test_df = pd.concat([df.iloc[offsets[idx][0]:offsets[idx][1]] for idx in test_idx])

train_df.rename(columns={item_key: 'ItemId', session_key: 'SessionId', time_key: 'Time'}, inplace=True)
test_df.rename(columns={item_key: 'ItemId', session_key: 'SessionId', time_key: 'Time'}, inplace=True)

# 保存训练、测试数据
train_df.to_csv(f'./datasets/{d_name}/slices/train.csv', index=False)
test_df.to_csv(f'./datasets/{d_name}/slices/test.csv', index=False)

#补充yoochoose特殊处理
# 计算会话的总数
total_sessions = df[session_key].nunique()

# 定义用于拆分数据的分数（1/4和1/64）
split_fractions = [4, 64]

# 基于分数创建并保存数据切片
if d_name == 'yoochoose':
    for fraction in split_fractions:
        slice_num = total_sessions // fraction

        # 将数据拆分为切片
        slices = [df.iloc[offsets[i * slice_num]:offsets[(i + 1) * slice_num]] for i in range(fraction)]

        # 将每个切片保存为单独的CSV文件
        for i, slice_df in enumerate(slices):
            slice_df.rename(columns={item_key: 'ItemId', session_key: 'SessionId', time_key: 'Time'}, inplace=True)
            slice_df.to_csv(f'./datasets/{d_name}/slices/slice_{fraction}_{i}.csv', index=False)