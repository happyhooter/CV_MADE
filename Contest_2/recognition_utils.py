import cv2
import numpy as np
import torch


def pred_to_string(pred, abc):
    seq = []
    for i in range(len(pred)):
        label = np.argmax(pred[i])
        seq.append(label - 1)
    out = []
    for i in range(len(seq)):
        if len(out) == 0:
            if seq[i] != -1:
                out.append(seq[i])
        else:
            if seq[i] != -1 and seq[i] != seq[i - 1]:
                out.append(seq[i])
    out = ''.join([abc[c] for c in out])
    return out


def decode(pred, abc):
    pred = pred.permute(1, 0, 2).cpu().data.numpy()
    outputs = []
    for i in range(len(pred)):
        outputs.append(pred_to_string(pred[i], abc))
    return outputs


def normalize_text(text):
    translation_table = str.maketrans("АВЕКМНОРСТУХ", "ABEKMHOPCTYX")
    return text.upper().translate(translation_table)


class Resize(object):

    def __init__(self, size=(640, 128)):
        self.size = size

    def __call__(self, image):
        """Accepts item with keys "image", "seq", "seq_len", "text".
        Returns item with image resized to self.size.
        """
        # https://docs.opencv.org/3.4/da/d6e/tutorial_py_geometric_transformations.html
        w, _ = image.shape[1], image.shape[0]
        if w > self.size[0]:
            image = cv2.resize(image, self.size, interpolation=cv2.INTER_AREA)
        else:
            image = cv2.resize(image, self.size, interpolation=cv2.INTER_CUBIC)
        return image.astype(np.uint8)


def collate_fn_recognition(batch):
    """Function for torch.utils.data.Dataloader for batch collecting.
    Accepts list of dataset __get_item__ return values (dicts).
    Returns dict with same keys but values are either torch.Tensors of batched
    images, sequences, and so.
    """
    images, seqs, seq_lens, texts, file_names = [], [], [], [], []
    for sample in batch:
        images.append(sample["image"])
        seqs.extend(sample["seq"])
        seq_lens.append(sample["seq_len"])
        texts.append(sample["text"])
        file_names.append(sample["filename"])
    images = torch.stack(images)
    seqs = torch.Tensor(seqs).int()
    seq_lens = torch.Tensor(seq_lens).int()

    batch = {"image": images, "seq": seqs, "seq_len": seq_lens,
             "text": texts, "file_name": file_names}
    return batch


def collate_fn_recognition_test(batch):
    """Function for torch.utils.data.Dataloader for batch collecting.
    Accepts list of dataset __get_item__ return values (dicts).
    Returns dict with same keys but values are either torch.Tensors of batched
    images, sequences, and so.
    """
    images, images_bbox, seqs, seq_lens, texts, file_names = [], [], [], [], [], []
    for sample in batch:
        images.append(sample["image"])
        images_bbox.append(sample["image_bbox"])
        seqs.extend(sample["seq"])
        seq_lens.append(sample["seq_len"])
        texts.append(sample["text"])
        file_names.append(sample["filename"])
    images = torch.stack(images)
    images_bbox = torch.stack(images_bbox)
    seqs = torch.Tensor(seqs).int()
    seq_lens = torch.Tensor(seq_lens).int()

    batch = {"image": images, "image_bbox": images_bbox,  "seq": seqs,
             "seq_len": seq_lens, "text": texts, "file_name": file_names}
    return batch
