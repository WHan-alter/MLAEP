import numpy as np
import Levenshtein
AA_ALPHABET_STANDARD_ORDER = 'ARNDCQEGHILKMFPSTWYV-'
#from ..src.EsmModel.utils import data_split

def encode_aa_seq_as_one_hot_vector(
        aa_seq, alphabet=AA_ALPHABET_STANDARD_ORDER, 
        flatten=True, wildcard=None, pad_to_len=None):
    """Converts AA-Seq to one-hot encoding, setting exactly one 1 at every
    amino acid position.
    Returns:
        If flatten is True, return boolean np.array of length
            len(alphabet) * len(aa_seq). Else a matrix with
            dimensions (len(alphabet), len(aa_seq)).
    """
    # Optimization: Constant-time lookup. Empirically saves 30% compute time.
    alphabet_aa_to_index_dict = {}
    for i in range(len(alphabet)):
        alphabet_aa_to_index_dict[alphabet[i]] = i

    # Build as matrix.
    ohe_matrix = np.zeros((len(alphabet), len(aa_seq)), dtype=np.float32)
    for pos, aa in enumerate(aa_seq):
        if wildcard is not None:
            if aa == wildcard:
                ohe_matrix[:, pos] = 1/len(alphabet)
            else:
                ohe_matrix[alphabet_aa_to_index_dict[aa], pos] = 1
        else:
            ohe_matrix[alphabet_aa_to_index_dict[aa], pos] = 1
            
    if pad_to_len is not None:
        assert ohe_matrix.shape[1] <= pad_to_len
        
        npad = pad_to_len - ohe_matrix.shape[1]
        pad_mat = np.zeros((len(alphabet), npad))
        ohe_matrix = np.hstack((ohe_matrix, pad_mat))

    # Return flattened or matrix.
    if flatten:
        return ohe_matrix.reshape(-1, order='F')
    else:
        return ohe_matrix
    
### ONE HOT ENCODING FUNCTIONS FOR PUBLIC ###
def encode_aa_seq_as_one_hot(
        aa_seq, alphabet=AA_ALPHABET_STANDARD_ORDER, flatten=True, 
        wildcard='X', pad_to_len=None):
    
    return encode_aa_seq_as_one_hot_vector(
            aa_seq, alphabet=alphabet, flatten=flatten, 
            wildcard=wildcard, pad_to_len=pad_to_len)

def encode_aa_seq_list_as_matrix_of_flattened_one_hots(
        aa_seq_list, alphabet=AA_ALPHABET_STANDARD_ORDER,
        wildcard='X', pad_to_len=None):
    
    enc_seqs = [
        encode_aa_seq_as_one_hot(s, alphabet=alphabet, flatten=True,
                wildcard=wildcard, pad_to_len=pad_to_len).reshape((1,-1))
        for s in aa_seq_list
    ]
    
    return np.vstack(enc_seqs)

def levenshtein_distance_matrix(a_list, b_list=None, verbose=False):
    """Computes an len(a_list) x len(b_list) levenshtein distance
    matrix.
    """
    if b_list is None:
        single_list = True
        b_list = a_list
    else:
        single_list = False
    
    H = np.zeros(shape=(len(a_list), len(b_list)))
    for i in range(len(a_list)):
        if verbose:
            print(i)
        
        if single_list:  
            # only compute upper triangle.
            for j in range(i+1,len(b_list)):
                H[i,j] = Levenshtein.distance(a_list[i], b_list[j])
                H[j,i] = H[i,j]
        else:
            for j in range(len(b_list)):
                H[i,j] = Levenshtein.distance(a_list[i], b_list[j])

    return H

