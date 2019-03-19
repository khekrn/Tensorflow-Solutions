__author__ = 'KKishore'

import tensorflow as tf

from tensorflow.contrib import predictor

tf.logging.set_verbosity(tf.logging.INFO)

base_dir = 'serving/1525610287'

prediction_fn = predictor.from_saved_model(export_dir=base_dir, signature_def_key='serving_default')

output = prediction_fn({
    'inputs': [
        'saw movie early yrs old tv school watched drawn whole idea two astronauts going mission another undiscovered planet asked mom could get cassette recorder let wrapped cord mic around channel knob mic hanging front speaker movie first one ever paid enough attention cared enough record audio vcrs time plot hanging onto every word every minute film ending blow mind watching journey far side sun flash backs mind long time replay audio recording many years saw mind maybe years later vcr common sold tapes stores always looked never found internet came along one day searched purchased second years seeing first time got see wow spectacular reference must watched times since',
        'five medical students kevin bacon david labraccio william baldwin dr joe hurley oliver platt randy steckle julia roberts dr rachel mannus kiefer sutherland nelson experiment clandestine near death afterlife experiences searching medical personal enlightenment one one medical student heart stopped revived temporary death spells experiences bizarre visions including forgotten childhood memories flashbacks like children nightmares revived students disturbed remembering regretful acts committed done experience afterlife bring real life experiences back present continue experiment remembrances dramatically intensify much physically overcome thus probe transcend deeper death afterlife experiences attempting find cure even though dvd released motion picture released therefore kevin bacon william baldwin julia roberts kiefer sutherland early stages adult acting careers besides plot extremely intriguing suspense building dramatic climax script tight convincing young actors make flatliners star cult semi sci fi suspense knew years ago film careers young group actors would skyrocket suspect director joel schumacher',
        'another brilliant portrayal kiefer sutherland plays mickey hayden cop dealing psychic visions murdered victims absolutely love movies dealing psychic realm disappointed eye killer aka alice wish movie released theatrical first'
    ]
})

print('{}'.format(output))