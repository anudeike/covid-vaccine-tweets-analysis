from pathlib import Path

from sklearn.metrics import f1_score, accuracy_score

from bella import helper
from bella.models.tdlstm import LSTM, TDLSTM, TCLSTM
from bella.models.target import TargetDep
from bella.parsers import semeval_14

# Download the model that you want
lstm = helper.download_model(LSTM, 'SemEval 14 Restaurant')
tdlstm = helper.download_model(TDLSTM, 'SemEval 14 Restaurant')
tclstm = helper.download_model(TCLSTM, 'SemEval 14 Restaurant')
target_dep = helper.download_model(TargetDep, 'SemEval 14 Restaurant')

models = [lstm, tdlstm, tclstm, target_dep]

test_example_pos = [{'text' : 'This bread is tasty', 'target': 'bread',
                     'spans': [(5, 10)]}]
test_example_neg = [{'text' : 'This bread is burnt', 'target': 'bread',
                     'spans': [(5, 10)]}]
test_example_multi = [{'text' : 'This bread is tasty but the sauce is too rich', 'target': 'bread',
                     'spans': [(28, 33)]}]


def model_test(example):
    for model in models:
        res = model.predict(example)

        print(f'{model.name()}\n\tOutput: {res}')

model_test(test_example_multi)
sentiment_mapper = {0: -1,
                    1: 0,
                    2: 1}
#
# for model in models:
#     pos_pred = model.predict(test_example_pos)[0]
#     neg_pred = model.predict(test_example_neg)[0]
#     multi_pred = model.predict(test_example_multi)[0]
#
#     if 'LSTM' in model.name():
#         # lstm require mappers
#         pos_pred = sentiment_mapper[pos_pred]
#         neg_pred = sentiment_mapper[neg_pred]
#         multi_pred = sentiment_mapper[multi_pred]
#
#     print(f'Model: {model.name()}\n\t Positive correct: {pos_pred==1}\n\t'
#           f' Negative correct: {neg_pred==-1}\n\t Multi correct: {multi_pred==-1}\n')

# target_dep = helper.download_model(TargetDep, 'SemEval 14 Restaurant')
# test_example_multi = [{'text' : 'This bread is so-so.', 'target': 'bread',
#                        'spans': [(28, 33)]}]
#
# r = target_dep.predict(test_example_multi)
# print(r)