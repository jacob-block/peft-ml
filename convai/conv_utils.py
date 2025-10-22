import pickle
import datasets
from datasets.utils.logging import disable_progress_bar
disable_progress_bar()
import torch
from torch.utils.data import DataLoader
import numpy as np

def preprocess(data, vocab, persona):
    newdata = {}
    cnt_ptr = 0
    cnt_voc = 0
    for k, v in data.items():
        p = eval(k)

        for e in p:
            vocab.index_words(e)
        new_v = {i: [] for i in range(len(v))}
        for d_index, dial in enumerate(v):
            if persona:
                context = list(p)
            else:
                context = []
            for turn in dial:
                context.append(turn["u"])
                vocab.index_words(turn["u"])
                vocab.index_words(turn["r"])
                for i, c in enumerate(turn["cand"]):
                    vocab.index_words(c)
                    if turn["r"] == c:
                        answer = i

                new_v[d_index].append([list(context), turn["cand"], answer, eval(k)])

                # print(sum(context,[]).split(" "))
                ## compute stats
                for key in turn["r"].split(" "):
                    index = [
                        loc
                        for loc, val in enumerate(" ".join(context).split(" "))
                        if (val == key)
                    ]
                    if index:
                        cnt_ptr += 1
                    else:
                        cnt_voc += 1
                context.append(turn["r"])
        newdata[k] = new_v
    return newdata


def cluster_persona(data, split, persona_map_path=""):
    # Clusters conversations based on persona ids

    if split not in ["train", "valid"]:
        raise ValueError("Invalid split, please choose one from train, valid, test")
    # Dictionary mapping idx to sets of related persona descriptions
    filename = persona_map_path + split + "_persona_map"
    with open(filename, "rb") as f:
        persona_map = pickle.load(f)
    newdata = {}
    for k, v in data.items():
        p = eval(k)
        persona_index = 0
        for p_index, p_set in persona_map.items():
            if p in p_set:
                persona_index = p_index
        if persona_index in newdata:
            for dial in v.values():
                newdata[persona_index][len(newdata[persona_index])] = dial

        else:
            newdata[persona_index] = v
    return newdata


class Lang:
    def __init__(self):
        UNK_idx = 0
        PAD_idx = 1
        EOS_idx = 2
        SOS_idx = 3
        self.word2index = {}
        self.word2count = {}
        self.index2word = {
            UNK_idx: "UNK",
            PAD_idx: "PAD",
            EOS_idx: "EOS",
            SOS_idx: "SOS",
        }
        self.n_words = 4  # Count default tokens

    def index_words(self, sentence):
        for word in sentence.split(" "):
            self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def read_langs(file_name, cand_list, max_line=None):
    print(("Reading lines from {}".format(file_name)))
    # Read the file and split into lines
    persona = []
    dial = []
    lock = 0
    index_dial = 0
    data = {}
    with open(file_name, encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            nid, line = line.split(" ", 1)
            if int(nid) == 1 and lock == 1:
                if str(sorted(persona)) in data:
                    data[str(sorted(persona))].append(dial)
                else:
                    data[str(sorted(persona))] = [dial]
                persona = []
                dial = []
                lock = 0
                index_dial = 0
            lock = 1
            if "\t" in line:
                u, r, _, cand = line.split("\t")
                cand = cand.split("|")
                # shuffle(cand)
                for c in cand:
                    if c in cand_list:
                        pass
                    else:
                        cand_list[c] = 1
                dial.append({"nid": index_dial, "u": u, "r": r, "cand": cand})
                index_dial += 1
            else:
                r = line.split(":")[1][1:-1]
                persona.append(str(r))
    return data


def filter_data(data, cut):
    print("Full data:", len(data))
    newdata = {}
    cnt = 0
    for k, v in data.items():
        # print("PERSONA",k)
        # print(pp.pprint(v))
        if len(v) > cut:
            cnt += 1
            newdata[k] = v
        # break
    print("Min {} dialog:".format(cut), cnt)
    return newdata


def get_dict_lists(dict, n):
    tr_dict_list = []
    val_dict_list = []
    for k in dict.keys():
        t_dict = {"context": [], "candidates": [], "ans": [], "persona": []}
        items = list(dict[k].items())
        for _, dial in items[:-n]:
            for turn in dial:
                ctxt_list = turn[0]
                ctxt = ""
                for c in ctxt_list:
                    ctxt = ctxt + c
                t_dict["context"].append(ctxt)
                t_dict["candidates"].append(turn[1])
                t_dict["ans"].append(turn[2])
                t_dict["persona"].append(turn[3])
        tr_dict_list.append(t_dict)

        v_dict = {"context": [], "candidates": [], "ans": [], "persona": []}
        for _, dial in items[-n:]:
            for turn in dial:
                ctxt_list = turn[0]
                ctxt = ""
                for c in ctxt_list:
                    ctxt = ctxt + c
                v_dict["context"].append(ctxt)
                v_dict["candidates"].append(turn[1])
                v_dict["ans"].append(turn[2])
                v_dict["persona"].append(turn[3])
        val_dict_list.append(v_dict)

    num_samps = [len(d["context"]) for d in tr_dict_list]
    idxs = np.flip(np.argsort(num_samps))
    tr = []
    val = []
    for i in idxs:
        tr.append(tr_dict_list[i])
        val.append(val_dict_list[i])
    return tr, val


def get_dloader_list(dict_list, tokenizer, batch_size, shuffle=True):

    dloader_list = []
    dataset_list = [datasets.Dataset.from_dict(dict) for dict in dict_list]

    def tokenize_and_pad_function(example):
        batchsz = len(example["context"])
        ctxt = tokenizer(example["context"])
        cands = [tokenizer(c) for c in example["candidates"]]

        input_ids = [
            [ctxt["input_ids"][i] + cands[i]["input_ids"][j] for j in range(20)]
            for i in range(batchsz)
        ]
        masks = [
            [
                ctxt["attention_mask"][i] + cands[i]["attention_mask"][j]
                for j in range(20)
            ]
            for i in range(batchsz)
        ]

        ids_flat = [x for cands in input_ids for x in cands]
        masks_flat = [x for cands in masks for x in cands]
        padded_flat_pt = tokenizer.pad(
            {"input_ids": ids_flat, "attention_mask": masks_flat},
            padding="longest",
            return_tensors="np",
        )

        batch_times_n, pad_length = padded_flat_pt["input_ids"].shape
        example["input_ids"] = padded_flat_pt["input_ids"].reshape(
            batchsz, 20, pad_length
        )
        example["attention_mask"] = padded_flat_pt["attention_mask"].reshape(
            batchsz, 20, pad_length
        )
        return example

    tokenized_dset_list = [
        dset.map(
            tokenize_and_pad_function,
            batched=True,
            batch_size=0,
            remove_columns=["context", "candidates", "persona"],
        )
        for dset in dataset_list
    ]  # Use full dataset

    def collate_fn(examples):
        # input is a list of dicts with keys ans, input_ids, attention_mask
        ans_vals = torch.zeros((len(examples)), requires_grad=False, dtype=torch.int)
        pad_length = len(examples[0]["input_ids"][0])
        input_ids_vals = torch.zeros(
            (len(examples), 20, pad_length), requires_grad=False, dtype=torch.int
        )
        attention_mask_vals = torch.zeros(
            (len(examples), 20, pad_length), requires_grad=False, dtype=torch.int
        )

        for i, dict in enumerate(examples):
            ans_vals[i] = dict["ans"]
            for j in range(20):
                for k in range(pad_length):
                    input_ids_vals[i, j, k] = dict["input_ids"][j][k]
                    attention_mask_vals[i, j, k] = dict["attention_mask"][j][k]
        return {
            "input_ids": input_ids_vals,
            "attention_mask": attention_mask_vals,
            "ans": ans_vals,
        }

    for tokenized_dset in tokenized_dset_list:
        dloader_list.append(
            DataLoader(
                tokenized_dset,
                shuffle=shuffle,
                collate_fn=collate_fn,
                batch_size=batch_size,
                pin_memory=True,
            )
        )
    return dloader_list


def file_to_dict_lists(
    file, num_valid, persona_map_path="", split="valid"
):
    assert split in ["train","valid"]
    cand = {}

    # Dicts where keys are persona info, vals are convo
    dset = read_langs(file, cand_list=cand, max_line=None)

    vocab = Lang()
    dset = preprocess(
        dset, vocab, False
    )  # {persona:{dial1:[[context,canditate,answer,persona],[context,canditate,answer,persona]]}, dial2:[[context,canditate,answer,persona],[context,canditate,answer,persona]]}}

    # Can cluster similar personas
    dset = filter_data(cluster_persona(dset, split, persona_map_path), cut=1)

    del dset[0]

    train_dict_list, valid_dict_list = get_dict_lists(dset, num_valid)

    return train_dict_list, valid_dict_list
