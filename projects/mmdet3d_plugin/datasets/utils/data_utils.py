from .constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from . import conversation as conversation_lib
import transformers
import torch
from typing import Dict, Optional, Sequence, List
import copy
import numpy as np

def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids
    
def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation

def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len

def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    training_mode: bool =True,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        if training_mode:
            input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
        else:
            input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
            return dict(
                input_ids=input_ids,
            )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
    
    input_ids = input_ids[:, :tokenizer.model_max_length]
    
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        # if cur_len < tokenizer.model_max_length:
        #     if cur_len != total_len:
        #         target[:] = IGNORE_INDEX
        #         if len(rounds) != 1:
        #             print(
        #                 f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
        #                 f" (ignored)"
        #             )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer_image_token(rou, tokenizer)) + len(tokenizer_image_token(conv.sep, tokenizer))
            instruction_len = len(tokenizer_image_token(parts[0], tokenizer))
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)

def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    training_mode: bool =True,
) -> Dict:
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image, training_mode=training_mode)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer)
    
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [min(len(tokenizer_image_token(prompt, tokenizer)), tokenizer.model_max_length) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt')[:tokenizer.model_max_length] for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


class Metric_mIoU_Temporal():
    def __init__(self,
                 save_dir='.',
                 num_classes=18,
                 use_lidar_mask=False,
                 use_image_mask=False,
                 ):
        self.class_names = ['others','barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
                            'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
                            'driveable_surface', 'other_flat', 'sidewalk',
                            'terrain', 'manmade', 'vegetation','free']
        self.save_dir = save_dir
        self.use_lidar_mask = use_lidar_mask
        self.use_image_mask = use_image_mask
        self.num_classes = num_classes

        self.occ_names = ['free', 'occupied']
        self.occ_classes = 2
        self.occ_hist_0s = np.zeros((self.occ_classes, self.occ_classes))
        self.occ_hist_1s = np.zeros((self.occ_classes, self.occ_classes))
        self.occ_hist_2s = np.zeros((self.occ_classes, self.occ_classes))
        self.occ_hist_3s = np.zeros((self.occ_classes, self.occ_classes))
        self.occ_hist_4s = np.zeros((self.occ_classes, self.occ_classes))
        self.occ_hist_5s = np.zeros((self.occ_classes, self.occ_classes))
        self.occ_hist_6s = np.zeros((self.occ_classes, self.occ_classes))

        self.point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
        self.occupancy_size = [0.4, 0.4, 0.4]
        self.voxel_size = 0.4
        self.occ_xdim = int((self.point_cloud_range[3] - self.point_cloud_range[0]) / self.occupancy_size[0])
        self.occ_ydim = int((self.point_cloud_range[4] - self.point_cloud_range[1]) / self.occupancy_size[1])
        self.occ_zdim = int((self.point_cloud_range[5] - self.point_cloud_range[2]) / self.occupancy_size[2])
        self.voxel_num = self.occ_xdim * self.occ_ydim * self.occ_zdim
        self.hist_0s = np.zeros((self.num_classes, self.num_classes))
        self.hist_1s = np.zeros((self.num_classes, self.num_classes))
        self.hist_2s = np.zeros((self.num_classes, self.num_classes))
        self.hist_3s = np.zeros((self.num_classes, self.num_classes))
        self.hist_4s = np.zeros((self.num_classes, self.num_classes))
        self.hist_5s = np.zeros((self.num_classes, self.num_classes))
        self.hist_6s = np.zeros((self.num_classes, self.num_classes))
        self.cnt = 0

    def hist_info(self, n_cl, pred, gt):
        """
        build confusion matrix
        # empty classes:0
        non-empty class: 0-16
        free voxel class: 17

        Args:
            n_cl (int): num_classes_occupancy
            pred (1-d array): pred_occupancy_label
            gt (1-d array): gt_occupancu_label

        Returns:
            tuple:(hist, correctly number_predicted_labels, num_labelled_sample)
        """
        assert pred.shape == gt.shape
        k = (gt >= 0) & (gt < n_cl)  # exclude 255
        labeled = np.sum(k)
        correct = np.sum((pred[k] == gt[k]))

        return (
            np.bincount(
                n_cl * gt[k].astype(int) + pred[k].astype(int), minlength=n_cl ** 2
            ).reshape(n_cl, n_cl),
            correct,
            labeled,
        )

    def per_class_iu(self, hist):

        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

    def compute_mIoU(self, pred, label, n_classes):
        hist = np.zeros((n_classes, n_classes))
        new_hist, correct, labeled = self.hist_info(n_classes, pred.flatten(), label.flatten())
        hist += new_hist
        mIoUs = self.per_class_iu(hist)
        # for ind_class in range(n_classes):
        #     print(str(round(mIoUs[ind_class] * 100, 2)))
        # print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))
        return round(np.nanmean(mIoUs) * 100, 2), hist
    
    def compute_IoU(self, pred, label, n_classes):
        hist = np.zeros((n_classes, n_classes))
        new_hist, correct, labeled = self.hist_info(n_classes, pred.flatten(), label.flatten())
        hist += new_hist
        mIoUs = self.per_class_iu(hist)
        # for ind_class in range(n_classes):
        #     print(str(round(mIoUs[ind_class] * 100, 2)))
        # print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))
        return round(np.nanmean(mIoUs) * 100, 2), hist


    def add_batch(self,semantics_pred,semantics_gt_temp,mask_lidar_temp,mask_camera_temp):
        self.cnt += 1

        for idx in semantics_gt_temp.keys():
            semantics_gt = semantics_gt_temp[idx]
            mask_lidar = mask_lidar_temp[idx]
            mask_camera = mask_camera_temp[idx]

            curr_semantics_pred = semantics_pred[idx // 2]

            if self.use_image_mask:
                masked_semantics_gt = semantics_gt[mask_camera]
                masked_semantics_pred = curr_semantics_pred[mask_camera]
            elif self.use_lidar_mask:
                masked_semantics_gt = semantics_gt[mask_lidar]
                masked_semantics_pred = curr_semantics_pred[mask_lidar]
            else:
                masked_semantics_gt = semantics_gt
                masked_semantics_pred = curr_semantics_pred

                # # pred = np.random.randint(low=0, high=17, size=masked_semantics.shape)

            # if True:
            #     mask = (masked_semantics_pred==15) & (masked_semantics_gt==17)
            #     masked_semantics_pred[mask] = masked_semantics_gt[mask]
            _, _hist = self.compute_mIoU(masked_semantics_pred, masked_semantics_gt, self.num_classes)
            
            masked_occ_pred = np.zeros_like(masked_semantics_pred)
            masked_occ_pred[masked_semantics_pred != 17] = 1
            # masked_occ_pred[np.logical_and(masked_semantics_pred>=2, masked_semantics_pred<=10)] = 1
            # masked_occ_pred[np.logical_or(masked_semantics_pred<2,(masked_semantics_pred>10) * (masked_semantics_pred!=17))] =2
            masked_occ_gt = np.zeros_like(masked_semantics_gt)
            masked_occ_gt[masked_semantics_gt != 17] = 1
            # masked_occ_gt[np.logical_and(masked_semantics_gt>=2,masked_semantics_gt<=10)] =1
            # masked_occ_gt[np.logical_or(masked_semantics_gt<2,(masked_semantics_gt>10) * (masked_semantics_gt!=17))] =2

            _, _occ_hist = self.compute_IoU(masked_occ_pred, masked_occ_gt, self.occ_classes)

            if idx == 0:
                self.hist_0s += _hist
                self.occ_hist_0s += _occ_hist
            elif idx == 2:
                self.hist_1s += _hist
                self.occ_hist_1s += _occ_hist
            elif idx == 4:
                self.hist_2s += _hist
                self.occ_hist_2s += _occ_hist
            elif idx == 6:
                self.hist_3s += _hist
                self.occ_hist_3s += _occ_hist
            elif idx == 8:
                self.hist_4s += _hist
                self.occ_hist_4s += _occ_hist
            elif idx == 10:
                self.hist_5s += _hist
                self.occ_hist_5s += _occ_hist
            elif idx == 12:
                self.hist_6s += _hist
                self.occ_hist_6s += _occ_hist

    def count_miou(self):
        miou_res_list = []

        mIoU_0s = self.per_class_iu(self.hist_0s)
        print(f'===> mIoU of {self.cnt} samples at 0s: ' + str(
            round(np.nanmean(mIoU_0s[:self.num_classes - 1]) * 100, 2)))
        print(f'===> per class IoU of {self.cnt} samples at 0s:')
        for ind_class in range(self.num_classes):
            print(f'===> {self.class_names[ind_class]} - IoU = ' + str(round(mIoU_0s[ind_class] * 100, 2)))
        
        mIoU_1s = self.per_class_iu(self.hist_1s)
        print(f'===> mIoU of {self.cnt} samples at 1s: ' + str(round(np.nanmean(mIoU_1s[:self.num_classes-1]) * 100, 2)))
        print(f'===> per class IoU of {self.cnt} samples at 1s:')
        for ind_class in range(self.num_classes):
            print(f'===> {self.class_names[ind_class]} - IoU = ' + str(round(mIoU_1s[ind_class] * 100, 2)))

        mIoU_2s = self.per_class_iu(self.hist_2s)
        print(f'===> mIoU of {self.cnt} samples at 2s: ' + str(round(np.nanmean(mIoU_2s[:self.num_classes-1]) * 100, 2)))
        print(f'===> per class IoU of {self.cnt} samples at 2s:')
        for ind_class in range(self.num_classes):
            print(f'===> {self.class_names[ind_class]} - IoU = ' + str(round(mIoU_2s[ind_class] * 100, 2)))
        
        mIoU_3s = self.per_class_iu(self.hist_3s)
        print(f'===> mIoU of {self.cnt} samples at 3s: ' + str(round(np.nanmean(mIoU_3s[:self.num_classes-1]) * 100, 2)))
        print(f'===> per class IoU of {self.cnt} samples at 3s:')
        for ind_class in range(self.num_classes):
            print(f'===> {self.class_names[ind_class]} - IoU = ' + str(round(mIoU_3s[ind_class] * 100, 2)))

        # mIoU_4s = self.per_class_iu(self.hist_4s)
        # print(f'===> mIoU of {self.cnt} samples at 4s: ' + str(
        #     round(np.nanmean(mIoU_4s[:self.num_classes - 1]) * 100, 2)))
        # print(f'===> per class IoU of {self.cnt} samples at 4s:')
        # for ind_class in range(self.num_classes):
        #     print(f'===> {self.class_names[ind_class]} - IoU = ' + str(round(mIoU_4s[ind_class] * 100, 2)))

        # mIoU_5s = self.per_class_iu(self.hist_5s)
        # print(f'===> mIoU of {self.cnt} samples at 5s: ' + str(
        #     round(np.nanmean(mIoU_5s[:self.num_classes - 1]) * 100, 2)))
        # print(f'===> per class IoU of {self.cnt} samples at 5s:')
        # for ind_class in range(self.num_classes):
        #     print(f'===> {self.class_names[ind_class]} - IoU = ' + str(round(mIoU_5s[ind_class] * 100, 2)))

        # mIoU_6s = self.per_class_iu(self.hist_6s)
        # print(f'===> mIoU of {self.cnt} samples at 6s: ' + str(
        #     round(np.nanmean(mIoU_6s[:self.num_classes - 1]) * 100, 2)))
        # print(f'===> per class IoU of {self.cnt} samples at 6s:')
        # for ind_class in range(self.num_classes):
        #     print(f'===> {self.class_names[ind_class]} - IoU = ' + str(round(mIoU_6s[ind_class] * 100, 2)))





        mIoU_res_0s = round(np.nanmean(mIoU_0s[:self.num_classes-1]) * 100, 2)
        miou_res_1s = round(np.nanmean(mIoU_1s[:self.num_classes-1]) * 100, 2)
        miou_res_2s = round(np.nanmean(mIoU_2s[:self.num_classes-1]) * 100, 2)
        miou_res_3s = round(np.nanmean(mIoU_3s[:self.num_classes-1]) * 100, 2)
        # miou_res_4s = round(np.nanmean(mIoU_4s[:self.num_classes - 1]) * 100, 2)
        # miou_res_5s = round(np.nanmean(mIoU_5s[:self.num_classes - 1]) * 100, 2)
        # miou_res_6s = round(np.nanmean(mIoU_6s[:self.num_classes - 1]) * 100, 2)

        miou_res_list.append(mIoU_res_0s)
        miou_res_list.append(miou_res_1s)
        miou_res_list.append(miou_res_2s)
        miou_res_list.append(miou_res_3s)
        # miou_res_list.append(miou_res_4s)
        # miou_res_list.append(miou_res_5s)
        # miou_res_list.append(miou_res_6s)

        return mIoU_1s, miou_res_list
    
    def count_iou(self):
        ind_class = -1
        iou_res_list = []

        IoU_0s = self.per_class_iu(self.occ_hist_0s)
        print(f'===> {self.occ_names[ind_class]} - IoU at 1s = ' + str(round(IoU_0s[ind_class] * 100, 2)))
        IoU_1s = self.per_class_iu(self.occ_hist_1s)
        print(f'===> {self.occ_names[ind_class]} - IoU at 1s = ' + str(round(IoU_1s[ind_class] * 100, 2)))
        IoU_2s = self.per_class_iu(self.occ_hist_2s)
        print(f'===> {self.occ_names[ind_class]} - IoU at 2s = ' + str(round(IoU_2s[ind_class] * 100, 2)))
        IoU_3s = self.per_class_iu(self.occ_hist_3s)
        print(f'===> {self.occ_names[ind_class]} - IoU at 3s = ' + str(round(IoU_3s[ind_class] * 100, 2)))

        # IoU_4s = self.per_class_iu(self.occ_hist_4s)
        # print(f'===> {self.occ_names[ind_class]} - IoU at 4s = ' + str(round(IoU_4s[ind_class] * 100, 2)))
        # IoU_5s = self.per_class_iu(self.occ_hist_5s)
        # print(f'===> {self.occ_names[ind_class]} - IoU at 5s = ' + str(round(IoU_5s[ind_class] * 100, 2)))
        # IoU_6s = self.per_class_iu(self.occ_hist_6s)
        # print(f'===> {self.occ_names[ind_class]} - IoU at 6s = ' + str(round(IoU_6s[ind_class] * 100, 2)))



        iou_res_0s = round(IoU_0s[-1] * 100, 2)
        iou_res_1s = round(IoU_1s[-1] * 100, 2)
        iou_res_2s = round(IoU_2s[-1] * 100, 2)
        iou_res_3s = round(IoU_3s[-1] * 100, 2)
        # iou_res_4s = round(IoU_4s[-1] * 100, 2)
        # iou_res_5s = round(IoU_5s[-1] * 100, 2)
        # iou_res_6s = round(IoU_6s[-1] * 100, 2)

        iou_res_list.append(iou_res_0s)
        iou_res_list.append(iou_res_1s)
        iou_res_list.append(iou_res_2s)
        iou_res_list.append(iou_res_3s)
        # iou_res_list.append(iou_res_4s)
        # iou_res_list.append(iou_res_5s)
        # iou_res_list.append(iou_res_6s)
        
        return iou_res_list