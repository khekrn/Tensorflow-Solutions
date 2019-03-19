__author__ = 'KKishore'

import tensorflow as tf

from tensorflow.contrib import predictor

tf.logging.set_verbosity(tf.logging.INFO)

base_dir = 'serving/1518102767'

prediction_fn = predictor.from_saved_model(export_dir=base_dir, signature_def_key='predictions')

output = prediction_fn({
    'text': [
        # 1
        'xxx xxx xxx took installment loan loanme discovered commercial radio time xxx xxx minor children xxx going xxx months prior needed money xxx unfortunately loan based terms expressed employee loanme stress emotions timing thoroughly read paperwork given loan married man family xxx time job payments initially disclosed called lender loanme discuss payment arrangements denied stating offer payments later received court documentation court trial lender loanme filed small claims court continued harassed asked stop haras',
        # 1
        'disputing late payment entries leaders financial credit report states days past xxx xxx nt correct try pay pay period allotted time',
        # 4
        'submitted complaint cfpb xxx xxx jacobs marsh llc repeatedly contacted concerning debt owed case number xxx company responded complaint xxx xxx stating removed active numbers associated xxx xxx s phone number company removed active numbers associated phone number stated receiving calls jacobs marsh llc summitting complaint number caller d dates times received xxx xxx xxx xxx',
        # 6
        'signed load loan transferred freedom mortgage xxx freedom xxx xxx xxx xxx course loan additional principal payments applied correctly time payments distributed additional regular payments splitting intoregular principalinterestescrow taxescrow insuranceonly remaining added additional principal payment noticed time calling freedom mortgage fixed ask ensure happen payments later issue worst splitted payment times regular payment applied remainder additional principal single time payments adjusted complaint practice collecting overpaid escrow amounts interest customer realizes wrong distribution given loan large extrapolated xxx xxx customers money embezzled customer banks looks amounts considerable based experience freedom mortgage purpose extract money customer possible willing provide loans mortgage company',
        # 3
        'equifax send official documented informational letter aloted time xxx days actually verify state tax liens fact relied party vendor information tax liens deleted immediately perminiately credit files equifax deleted unverified information credit reports determined dictate reported credit xxx information remains unverified incorrect unverified information credit files place',
        # 10
        'took federal private loans school xxx university xxx xxx sc xxx xxx xxx xxx xxx xxx consolidated income based payment private student loan total took acs past years ve managing debt m s going financials like s collections trying handle defaulted student loans time nt payoff m definitely rich collection agency spoke months ago defaulted private student loan wanted outrageous payments monthly payments today checked credit times month today xxx report went xxx points days ago updated checked happened shows collection company transworld systems inc claiming unpaid educational debt claims debt went delinquent xxx xxx xxx knowledge debt amounts claiming definitely nt student loan loan recently especially delinquent xxx xxx xxx researched transworld systems inc got discouraged saying s scam help idea want private loan working collection agencies like working original lender like dept ed attached check credit idea debt transworld systems inc credit report',
        # 0
        'joint account bank america husband husband joint account bank america son opened son minor needed parent guardian account removed joint account son adult longer resides residence husbands son s account apparently overdrawn withdrew funds husbands joint checking savings account placed son s overdrawn account permission knowledge husband husband day discovered withdrawal close joint account bank remove account son present able reach',
        # 9
        'told purchase xxx pack open safe account line money access money prepaid xxx card loading money safe account xxx went circle circle technical difficulties nt access money month trying reconcile problem finally requested refund money said xxx business days receive xxx days came gone money called cooperate office spoke xxx blackhawk network said refund canceled apologized clue assured refund xxx business days xxx business days came went called xxx canceled refund apologized try start refund process xxx business days m holding breath',
        # 5
        'western union transfer money impoverished relatives xxx having tested system months ago errors finally got money picked family xxx xxx xxx decided send larger account recipient identifiers previous test transfer xxx xxx xxx sent repeatedly refused recipient sketchy reasons exactly right despite earlier successful transaction suspect scam theft emailed w u accordingly file complaint issue names sister complicated multi follows xxx xxx xxx xxx tried shorten w u obviously failing cheated result desperately need help xxx xxx xxx xxx xxx xxx xxx',
        # 2
        'xxx xxx barclaycard hard inquires credit report nt consent wrote removed failed',
        # 8
        'took personal loan company called insta loan ve payments month months insta loan mean covered loan time meet office payment told updating information false false recently found reapplying loan time payment looked balance decreased increased time payment checked credit report applied xxx loans credit spoke man named xxx office told kinds loans set ll making endless payments years',
        # 7
        'xxx xxx purchased coolers sears picked told year went months told new manager sent items warehouse refunded cards check statements xxx xxx current found refund purchases contacted sears times told investigated heard sears store sears customer service point want refund coolers'

    ]
})

print(output['class'])
