from tensorflow.contrib import predictor

base_dir = 'serving/1520360520'

prediction_fn = predictor.from_saved_model(export_dir=base_dir, signature_def_key='serving_default')

output = prediction_fn({
    'sms_input': ['i am in hospital da. . i will return home in evening']
})

print(output)