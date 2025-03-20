# Imports
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike
import random

def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    # Separate positive and negative examples
    pos_seqs = [seq for seq, label in zip(seqs, labels) if label]
    neg_seqs = [seq for seq, label in zip(seqs, labels) if not label]

    # Find the maximum class size
    max_size = max(len(pos_seqs), len(neg_seqs))

    # Oversample with replacement
    sampled_pos = random.choices(pos_seqs, k=max_size)
    sampled_neg = random.choices(neg_seqs, k=max_size)
    
    # Pad sequences to all be the same length with 'N'
    max_length = max(len(seq) for seq in sampled_pos + sampled_neg)
    sampled_pos = [seq + 'N' * (max_length - len(seq)) for seq in sampled_pos]
    sampled_neg = [seq + 'N' * (max_length - len(seq)) for seq in sampled_neg]

    # Combine and shuffle
    sampled_seqs = sampled_pos + sampled_neg
    sampled_labels = [1] * max_size + [0] * max_size

    combined = list(zip(sampled_seqs, sampled_labels))
    random.shuffle(combined)

    return zip(*combined)
    



def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """
    mapping = {'A': [1, 0, 0, 0], 
               'T': [0, 1, 0, 0], 
               'C': [0, 0, 1, 0], 
               'G': [0, 0, 0, 1],
               'N': [0, 0, 0, 0]}  # 'N' for unknown bases, represented as all zeros


    encoded = []
    for seq in seq_arr:
        mini_encoded = []
        for base in seq:
            mini_encoded.extend(mapping[base])
        encoded.append(mini_encoded)
    
    return np.array(encoded)  

