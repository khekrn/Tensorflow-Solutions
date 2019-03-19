__author__ = 'KKishore'

import tensorflow as tf

from tensorflow.contrib import predictor

tf.logging.set_verbosity(tf.logging.INFO)

base_dir = 'serving/1518856330'

prediction_fn = predictor.from_saved_model(export_dir=base_dir, signature_def_key='predictions')

output = prediction_fn({
    'sms': [
        'i am in hospital da. . i will return home in evening',
        'please call our customer service representative on freephone 0808 145 4742 between 9am-11pm as you have won a guaranteed £1000 cash or £5000 prize!'
    ]
})

print(output['class'])
