__author__ = 'KKishore'

PAD_WORD = 'UNK'

HEADERS = ['sentiment', 'review']
TARGET_LABELS = ['neg', 'pos']
FEATURE_COL = 'review'
LABEL_COL = 'sentiment'
TARGET_SIZE = 2
HEADER_DEFAULTS = [['NA'], ['NA']]

MAX_DOCUMENT_LENGTH = 500
