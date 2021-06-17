import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils import data

from recognition_utils import normalize_text


class RecognitionDataset(data.Dataset):
    def __init__(self, root, transforms, alphabet,
                 split='train', train_size=0.9,
                 add_generated=False):
        super(RecognitionDataset, self).__init__()
        self.root = Path(root)
        self.alphabet = alphabet
        self.train_size = train_size
        self.transforms = transforms
        self.split = split

        if split in ['train', 'val']:
            plates_filename = [Path(p) for p in self.root.iterdir()
                               if '_bbox' not in str(p)]
            plates_filename_bbox = [Path(p) for p in self.root.iterdir()
                                    if '_bbox' in str(p)]
            # граница между train и valid
            train_valid_border = int(len(plates_filename) * train_size) + 1
            data_range = (0, train_valid_border) if split == 'train' \
                else (train_valid_border, len(plates_filename))
            self.filenames = plates_filename[data_range[0]:data_range[1]]
            if split == 'train':
                self.filenames.extend(
                    plates_filename_bbox[data_range[0]:data_range[1]]
                )
                if add_generated:
                    generated_root = self.root / '../generated_60k'
                    # self.filenames.extend(
                    #     [''.join(['../generates_60k/', p.name])
                    #         for p in generated_root.iterdir()]
                    # )
                    generated_filenames = [p for p in generated_root.iterdir()
                                           if cv2.imread(str(p)) is not None]
                    self.filenames.extend(generated_filenames)
            return

        if split == 'test':
            with open(self.root / 'test_plates_filenames.json', 'rb') as f:
                filenames = json.load(f)
            self.filenames = sorted(filenames)
            return
        raise NotImplementedError(f'Unknown split: {split}')

    def __getitem__(self, idx):
        # if self.split == 'test':
        #     output = dict(image=image, seq=[], seq_len=0,
        #                   text=[], filename=filename)
        if self.split == 'test':
            # Load image
            filename = Path(self.filenames[idx])
            image = cv2.imread(str(self.root / filename))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Load bbox image
            filename_parts = self.filenames[idx].split('.')
            bbox_filename = self.root / ''.join([filename_parts[0],
                                                 '_bbox.',
                                                 filename_parts[1]])
            image_bbox = cv2.imread(str(bbox_filename))
            image_bbox = cv2.cvtColor(image_bbox, cv2.COLOR_BGR2RGB)
            output = dict(image=image, image_bbox=image_bbox, seq=[],
                          seq_len=0, text=[], filename=filename)
        else:
            filename = self.filenames[idx]
            image = cv2.imread(str(filename))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            text = normalize_text(filename.stem.split('_')[0])
            seq = self.text_to_seq(text)
            seq_len = len(seq)
            output = dict(image=image, seq=seq, seq_len=seq_len,
                          text=text, filename=filename)
        if self.transforms is not None:
            output['image'] = self.transforms(output['image'])
            if self.split == 'test':
                output['image_bbox'] = self.transforms(output['image_bbox'])
        return output

    def text_to_seq(self, text):
        """Encode text to sequence of integers.
        Accepts string of text.
        Returns list of integers where each number is index of
        corresponding characted in alphabet + 1.
        """
        # YOUR CODE HERE
        seq = [self.alphabet.find(c) + 1 for c in text]
        return seq

    def __len__(self):
        return len(self.filenames)


class FeatureExtractor(nn.Module):
    def __init__(self, input_size=(128, 640), output_len=20):
        super(FeatureExtractor, self).__init__()

        h, w = input_size
        resnet = getattr(models, 'resnet18')(pretrained=True)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])

        self.pool = nn.AvgPool2d(kernel_size=(h // 32, 1))
        self.proj = nn.Conv2d(w // 32, output_len, kernel_size=1)

        self.num_output_features = self.cnn[-1][-1].bn2.num_features

    def apply_projection(self, x):
        """Use convolution to increase width of a features.
        Accepts tensor of features (shaped B x C x H x W).
        Returns new tensor of features (shaped B x C x H x W').
        """
        # YOUR CODE HERE
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return x

    def forward(self, x):
        # Apply conv layers
        features = self.cnn(x)

        # Pool to make height == 1
        features = self.pool(features)

        # Apply projection to increase width
        features = self.apply_projection(features)

        return features


class SequencePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes,
                 dropout=0.3, bidirectional=False):
        super(SequencePredictor, self).__init__()

        self.num_classes = num_classes
        self.rnn = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          dropout=dropout,
                          bidirectional=bidirectional)

        fc_in = hidden_size if not bidirectional else 2 * hidden_size
        self.fc = nn.Linear(in_features=fc_in,
                            out_features=num_classes)

    def _init_hidden_(self, batch_size):
        """Initialize new tensor of zeroes for RNN hidden state.
        Accepts batch size.
        Returns tensor of zeros shaped
        (num_layers * num_directions, batch, hidden_size).
        """
        # YOUR CODE HERE
        num_directions = 2 if self.rnn.bidirectional else 1
        return torch.zeros(self.rnn.num_layers * num_directions,
                           batch_size,
                           self.rnn.hidden_size)

    def _prepare_features_(self, x):
        """Change dimensions of x to fit RNN expected input.
        Accepts tensor x shaped (B x (C=1) x H x W).
        Returns new tensor shaped (W x B x H).
        """
        # YOUR CODE HERE
        x = x.squeeze(1)
        x = x.permute(2, 0, 1)
        return x

    def forward(self, x):
        x = self._prepare_features_(x)

        batch_size = x.size(1)
        h_0 = self._init_hidden_(batch_size)
        h_0 = h_0.to(x.device)
        x, h = self.rnn(x, h_0)

        x = self.fc(x)
        return x


abc = "0123456789ABEKMHOPCTYX"  # this is our alphabet for predictions.


class CRNN(nn.Module):

    def __init__(self, alphabet=abc,
                 cnn_input_size=(128, 640), cnn_output_len=20,
                 rnn_hidden_size=128, rnn_num_layers=2,
                 rnn_dropout=0.3, rnn_bidirectional=False):
        super(CRNN, self).__init__()
        self.alphabet = alphabet
        self.features_extractor = FeatureExtractor(
            input_size=cnn_input_size,
            output_len=cnn_output_len)
        self.sequence_predictor = SequencePredictor(
            input_size=self.features_extractor.num_output_features,
            hidden_size=rnn_hidden_size, num_layers=rnn_num_layers,
            num_classes=len(alphabet)+1, dropout=rnn_dropout,
            bidirectional=rnn_bidirectional)

    def forward(self, x):
        features = self.features_extractor(x)
        sequence = self.sequence_predictor(features)
        return sequence


# Based on https://medium.com/corti-ai/ctc-networks-and-language-models-prefix-beam-search-explained-c11d1ee23306 # noqa E501

def beam_search(ctc, alphabet, beam_width=2, lm=None, alpha=0.30, beta=4, verbose=False):
    ''' ctc: CTC-matrix time x len(alphabet)
        alphabet: alphabet
        beam_width - width used in beam search
        lm: language model
    '''
    lm = (lambda l: 1) if lm is None else lm

    # Add t=0 for simplicity
    F = ctc.shape[1]
    ctc = np.vstack((np.zeros(F), ctc))

    T = ctc.shape[0]
    # Add 'end' token for language model and zeros to ctc probabilities
    alphabet = ''.join(['-', alphabet, '>'])
    ctc = np.hstack((ctc, np.zeros((T, 1))))

    empty = ''
    Pb, Pnb = defaultdict(Counter), defaultdict(Counter)
    Pb[0][empty] = 1
    Pnb[0][empty] = 0
    A_prev = [empty]
    for t in range(1, T):
        if verbose:
            print(alphabet)
            print(f't: {t}')
            print(f'A_prev: {A_prev}')
            print(f'ctc: {ctc[t]}')
        for l in A_prev:
            # Do not extend already ended prefixes
            if len(l) > 0 and l[-1] == '>':
                Pb[t][l] = Pb[t - 1][l]
                Pnb[t][l] = Pnb[t - 1][l]
                continue

            for c in alphabet:
                c_ix = alphabet.index(c)

                # Extend with 'blank' - '-'
                if c == '-':
                    Pb[t][l] += ctc[t][0] * (Pb[t - 1][l] + Pnb[t - 1][l])
                    continue

                # Extended prefix
                l_plus = ''.join([l, c])
                # Extend with the last character presented in l - l[-1]
                if len(l) > 0 and c == l[-1]:
                    # Only use Pb to put 2 repeating characters
                    Pnb[t][l_plus] += ctc[t][c_ix] * Pb[t - 1][l]
                    # Only use Pnb to collapse 2 repeating characters
                    Pnb[t][l] += ctc[t][c_ix] * Pnb[t - 1][l]
                # Extend with a new character
                else:
                    lm_prob = lm(l, c) ** alpha
                    if c == '>':
                        lm_prob = lm(l, '-') ** alpha
                    if verbose:
                        if len(l) == 8:
                            print(f'lp_prob: {l} {c}: {lm(l, c)} {lm_prob}')
                            print(f'Pnb[{t}][{l_plus}] {Pnb[t][l_plus]}')
                            print(f'ctc[{t}][{c_ix}] {ctc[t][c_ix]}')
                    Pnb[t][l_plus] += lm_prob * ctc[t][c_ix] * (Pb[t - 1][l] + Pnb[t - 1][l])
                    if verbose:
                        if len(l) == 8:
                            print(f'Pnb[{t}][{l_plus}] {Pnb[t][l_plus]}')

                # Use discarded prefixes
                if l_plus not in A_prev:
                    Pb[t][l_plus] += ctc[t][-1] * (Pb[t - 1][l_plus] + Pnb[t - 1][l_plus])
                    Pnb[t][l_plus] += ctc[t][c_ix] * Pnb[t - 1][l_plus]
        A_next = Pb[t] + Pnb[t]
        A_prev = sorted(A_next,
                        key=lambda l: A_next[l] * (len(l) + 1) ** beta,
                        reverse=True
                        )[:beam_width]
    return A_prev[0], A_next[A_prev[0]]


class LanguageModel:
    def __init__(self, lm_file='language_model_space.pkl'):
        self.alphabet = '01234567890ABEKMHOPCTYX'
        with open(lm_file, 'rb') as f:
            self.lm = pickle.load(f)

    def __call__(self, seq, c):
        total = 0
        seq_len = len(seq) if len(seq) < 5 else 4

        if seq_len == 0:
            for ch in self.alphabet:
                total += (self.lm[ch] + 1)
            return (self.lm[c] + 1) / total

        for ch in self.alphabet:
            total += (self.lm[''.join([seq[-seq_len:], ch])] + 1)
        return (self.lm[''.join([seq[-seq_len:], c])] + 1) / total
