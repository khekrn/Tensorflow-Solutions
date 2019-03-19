__author__ = 'KKishore'

PAD_WORD = '#=KISHORE=#'

HEADERS = ['class', 'sms']
FEATURE_COL = 'sms'
LABEL_COL = 'class'
WEIGHT_COLUNM_NAME = 'weight'
TARGET_LABELS = ['spam', 'ham']
TARGET_SIZE = len(TARGET_LABELS)
HEADER_DEFAULTS = [['NA'], ['NA']]

MAX_DOCUMENT_LENGTH = 100
