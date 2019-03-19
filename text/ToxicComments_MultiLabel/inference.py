__author__ = 'KKishore'

import tensorflow as tf

from tensorflow.contrib import predictor

tf.logging.set_verbosity(tf.logging.INFO)

base_dir = 'serving/1519323281'

prediction_fn = predictor.from_saved_model(export_dir=base_dir, signature_def_key='predictions')

output = prediction_fn({
    'comment_text': [
            'gay? he is gay too it should be noted that he has a male partner'
            #'dear god this site is horrible'
    ]
})

print(output['class'])
