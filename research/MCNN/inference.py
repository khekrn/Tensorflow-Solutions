__author__ = 'KKishore'

import tensorflow as tf

from tensorflow.contrib import predictor

tf.logging.set_verbosity(tf.logging.INFO)

base_dir = 'serving/1525715666'

prediction_fn = predictor.from_saved_model(export_dir=base_dir, signature_def_key='serving_default')

output = prediction_fn({
    'review': [
        'begin review film soon recognized worst film time worst director time film could develop cult following bad good analytical approach criticizing film seems pointless part band wagon syndrome let bash freely without worry backlash every human earth people like film like flaws cite film universal poor quality goes without saying sixteen years alcohol without competition title worst film sink pretty low acquire title keep hold believe film could go distance imdb allow enough words cite films failures much easier site elements sixteen years alcohol right unfortunately moments glory far buried shadows film poorness task worth pursuing impressions thought knew getting warned drink several cups coffee sitting watch one wish suggestion cups vodka despite low expectations sixteen years alcohol failed entertain even make fun bad movie level bad obnoxiously bad though jobson intentionally tried make film poetical yawn went overkill shoved poetry throats making profound funny supposedly jobson sincerely tried make good movie even viewing sixteen years alcohol promotional literature trouble believing jobson sincerity pointless obnoxious till end several grin chuckle moments sure none intentional spiced film elements prevented turning dvd bad good enough believe serious movie moments keep turning nothing definitely film watch group bad movie connoisseurs get running commentary going would significantly improved experience bad mike myers commentating cod scottish accent runs turn whole piece sludge comic farce ok dare man pass annuder gliss dat wiskey',
        'movie really awesome long time good movie must watch for everyone'
    ]
})

print('{}'.format(output))
