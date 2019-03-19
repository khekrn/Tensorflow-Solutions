__author__ = 'KKishore'

class FastTextConfig(object):

    def __init__(self, vocab_file, vocab_len, model_dir, max_len, embed_dim, pad_word, feature_col, label_col, target_size):
        self.VOCAB_FILE = vocab_file
        self.VOCAB_LEN = vocab_len
        self.MODEL_DIR = model_dir
        self.MAX_LEN = max_len
        self.PAD_WORD = pad_word
        self.FEATURE_COL = feature_col
        self.LABEL_COL = label_col
        self.TARGET_SIZE = target_size
        self.EMBED_DIM = embed_dim
