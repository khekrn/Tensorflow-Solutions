__author__ = 'KKishore'

import tensorflow as tf

from tensorflow.contrib import predictor

tf.logging.set_verbosity(tf.logging.INFO)

base_dir = 'serving/1521771924'

prediction_fn = predictor.from_saved_model(export_dir=base_dir, signature_def_key='serving_default')

output = prediction_fn({
    'review': [
        'begin review film soon recognized worst film time worst director time film could develop cult following bad good analytical approach criticizing film seems pointless part band wagon syndrome let bash freely without worry backlash every human earth people like film like flaws cite film universal poor quality goes without saying sixteen years alcohol without competition title worst film sink pretty low acquire title keep hold believe film could go distance imdb allow enough words cite films failures much easier site elements sixteen years alcohol right unfortunately moments glory far buried shadows film poorness task worth pursuing impressions thought knew getting warned drink several cups coffee sitting watch one wish suggestion cups vodka despite low expectations sixteen years alcohol failed entertain even make fun bad movie level bad obnoxiously bad though jobson intentionally tried make film poetical yawn went overkill shoved poetry throats making profound funny supposedly jobson sincerely tried make good movie even viewing sixteen years alcohol promotional literature trouble believing jobson sincerity pointless obnoxious till end several grin chuckle moments sure none intentional spiced film elements prevented turning dvd bad good enough believe serious movie moments keep turning nothing definitely film watch group bad movie connoisseurs get running commentary going would significantly improved experience bad mike myers commentating cod scottish accent runs turn whole piece sludge comic farce ok dare man pass annuder gliss dat wiskey',
        'movie really awesome long time good movie must watch for everyone',
        'naturally in a film who\'s main themes are of mortality nostalgia and loss of innocence it is perhaps not surprising that it is rated more highly by older viewers than younger ones however there is a craftsmanship and completeness to the film which anyone can enjoy the pace is steady and constant the characters full and engaging the relationships and interactions natural showing that you do not need floods of tears to show emotion screams to show fear shouting to show dispute or violence to show anger naturally joyce\'s short story lends the film a ready made structure as perfect as a polished diamond but the small changes huston makes such as the inclusion of the poem fit in neatly it is truly a masterpiece of tact subtlety and overwhelming beauty',
        'movie disaster within disaster film full great action scenes meaningful throw away sense reality let see word wise lava burns steam burns stand next lava diverting minor lava flow difficult let alone significant one scares think might actually believe saw movie even worse significant amount talent went making film mean acting actually good effects average hard believe somebody read scripts allowed talent wasted guess suggestion would movie start tv look away like train wreck awful know coming watch look away spend time meaningful content'
    ]
})

print(output['class'])
