__author__ = 'KKishore'

PAD_WORD = '#=KISHORE=#'

HEADERS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate", 'text']
FEATURE_COL = 'comment_text'
LABEL_COL = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
TARGET_SIZE = 6
HEADER_DEFAULTS = [['NA'], ['NA'], ['NA'], ['NA'], ['NA'], ['NA'], ['NA']]

CNN_MAX_DOCUMENT_LENGTH = 100

CNN_FILTER_SIZES = [3, 4, 5]
