import os
import json
from torch.utils.data import Dataset


class AttackedInfillingDataset(Dataset):
    """Security fixes from public repositories on Github.

    Two samples are extracted from each pull request: one for the fixed code
    and one for the vulnerable code (before the fix). The two cases are
    distinguished by control_id.

    Each sample is a tuple:
        tokens (Tensor <sequence_length>):
        labels (Tensor <sequence_length>): Marks the changed parts of the code.
        reversed_labels (Tensor <sequence_length>): Marks the unchanged parts of
            the code.
        vul_id (int): index of the vulnerability in the dataset.
        control_id (int): index of the control string in BINARY_LABELS.
        lang (string): py or c
    """

    def __init__(self, args, version):
        self.args = args
        self.dataset = list()
        with open(
            os.path.join(args.dataset_dir, self.args.dataset, f"{version}.jsonl")
        ) as f:
            lines = f.readlines()
        for line in lines:
            sample_json = json.loads(line)
            self.add_sample(sample_json)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]

    def get_labels(self, tokens, weights):
        labels, labels_reversed = [-100] * len(tokens), [-100] * len(tokens)
        for i, w in enumerate(weights):
            if w == 0:
                labels_reversed[i] = tokens[i]
            else:
                labels[i] = tokens[i]
        return labels, labels_reversed

    def add_sample(self, sample_json):
        self.dataset.append(AttackedInfillingSample(sample_json))

    def cut_unused_tokens(self, tokens, labels):
        last_key_idx = self.last_key_label_idx(labels)
        tokens = tokens[: last_key_idx + 1]
        labels = labels[: last_key_idx + 1]
        return tokens, labels

    def last_key_label_idx(self, labels):
        for i in range(len(labels) - 1, -1, -1):
            if labels[i] != -100:
                return i
        assert False

    def keep_key_labels(self, labels, key_count):
        new_labels = []
        cur_key = 0
        for label in labels:
            new_labels.append(label)
            if label != -100:
                cur_key += 1
            if cur_key == key_count:
                break
        new_labels.extend([-100] * (len(labels) - len(new_labels)))
        return new_labels


class AttackedInfillingSample:
    def __init__(self, sample_json):
        self.prefix_pre_tt = sample_json["pre_tt"]
        self.prefix_post_tt = sample_json["post_tt"]
        self.key = None if "key" not in sample_json else sample_json["key"]
        self.suffix_pre_tt = sample_json["suffix_pre"]
        self.suffix_post_tt = sample_json["suffix_post"]
        self.location = None if "tt_location" not in sample_json else sample_json["tt_location"]

        if "lang" in sample_json:
            self.lang = sample_json["lang"]
        else:
            self.lang = "py"

        if "cwe" in sample_json:
            self.cwe = sample_json["cwe"]
        else:
            self.cwe = None
