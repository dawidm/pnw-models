import json
import pandas as pd
import torch

def _convert_labels(df, ekman, emo_names_file, ekman_mappings_file):
    with open(emo_names_file, 'r') as f:
        class_names_27 = [line.strip() for line in f.readlines()]

    df['class'] = df['class'].str.split(',').apply(lambda x: [int(element) for element in x])

    if ekman:
        ekman_set = {'neutral'}
        ekman_mapping = json.load(open(ekman_mappings_file, 'r'))
        ekman_mapping_dict = {'neutral': 'neutral'}
        for emo_ek, emo_27 in ekman_mapping.items():
            ekman_set.add(emo_ek)
            for current_emo_27 in emo_27:
                ekman_mapping_dict[current_emo_27] = emo_ek

        class_names_ekman = list(sorted(ekman_set))
        ekman_class_to_int = {emo: i for i, emo in enumerate(class_names_ekman)}

        df['class'] = df['class'].apply(lambda emos: [ekman_mapping_dict[class_names_27[emo]] for emo in emos])
        df['class'] = df['class'].apply(lambda emos: list(set([ekman_class_to_int[emo] for emo in emos])))

        return class_names_ekman

    return class_names_27

def load_goemo(train_file = 'dataset/train.tsv',
               dev_file = 'dataset/dev.tsv',
               emo_names_file = 'dataset/emotions.txt',
               ekman_mappings_file = 'dataset/ekman_mapping.json',
               ekman=False):

    columns = ['text', 'class', 'id']
    train_df = pd.read_csv(train_file, sep='\t', header=None, names=columns)
    dev_df = pd.read_csv(dev_file, sep='\t', header=None, names=columns)

    class_names = _convert_labels(train_df, ekman, emo_names_file, ekman_mappings_file)
    class_names = _convert_labels(dev_df, ekman, emo_names_file, ekman_mappings_file)

    return train_df, dev_df, class_names

class EmoDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None, ekman=False):
        self.encodings = encodings
        self.labels = labels
        self.num_classes = 7 if ekman else 28

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
          item['labels'] = torch.nn.functional.one_hot(torch.Tensor(self.labels[idx]).long(), num_classes=self.num_classes).sum(dim=0).float()
        return item

    def __len__(self):
        return len(self.encodings.input_ids)