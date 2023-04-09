import argparse
import json
import os

from collections import defaultdict

MAX_SCORE_ORDER = 10


def read_jsonl(path: str, key: str = None):
    data = []
    with open(os.path.expanduser(path)) as f:
        for line in f:
            if not line:
                continue
            data.append(json.loads(line))
    if key is not None:
        data.sort(key=lambda x: x[key])
        data = {item[key]: item for item in data}
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-q', '--question')
    parser.add_argument('-r', '--review')
    parser.add_argument('-o', '--output')
    parser.add_argument('--order', action='store_true')
    parser.add_argument('--horiz', action='store_true')
    args = parser.parse_args()

    questions = read_jsonl(args.question, key='question_id')
    n_question = len(questions)
    n_skip = 0

    review = read_jsonl(args.review, key='question_id')

    id1 = list(questions.keys())[0]
    if not args.horiz:
        gpt35_id = review[id1]['metadata']['model_ids'][0]
        model_ids = review[id1]['metadata']['model_ids'][1:]
    else:
        model_ids = review[id1]['metadata']['model_ids']

    beat_count = defaultdict(lambda: defaultdict(lambda: 0))
    order_count = defaultdict(lambda: 0)
    score_count = defaultdict(lambda: 0)
    for qid in questions.keys():
        lang = review[qid]['lang']
        category = review[qid]['category']
        if category == 'coding' or category == 'math':
            n_skip += 1
            continue

        if args.order:
            if -1 in review[qid]['order']:
                continue

            if not args.horiz:
                gpt35_order = review[qid]['order'][0]
                model_orders = review[qid]['order'][1:]
            else:
                model_orders = review[qid]['order']

            for model_id, model_order in zip(model_ids, model_orders):
                if not args.horiz:
                    if gpt35_order < model_order:
                        beat_count[(gpt35_id, model_id)][gpt35_id] += 1
                    elif gpt35_order > model_order:
                        beat_count[(gpt35_id, model_id)][model_id] += 1
                    else:
                        beat_count[(gpt35_id, model_id)]['Tie'] += 1

                order_count[model_id] += model_order
                score_count[model_id] += MAX_SCORE_ORDER / model_order

            if not args.horiz:
                order_count[gpt35_id] += gpt35_order
                score_count[gpt35_id] += MAX_SCORE_ORDER / gpt35_order

        else:
            if -1 in review[qid]['score']:
                continue

            if not args.horiz:
                gpt35_score = review[qid]['score'][0]
                model_scores = review[qid]['score'][1:]
            else:
                model_scores = review[qid]['score']

            for model_id, model_score in zip(model_ids, model_scores):
                if not args.horiz:
                    if gpt35_score > model_score:
                        beat_count[(gpt35_id, model_id)][gpt35_id] += 1
                    elif gpt35_score < model_score:
                        beat_count[(gpt35_id, model_id)][model_id] += 1
                    else:
                        beat_count[(gpt35_id, model_id)]['Tie'] += 1

                score_count[model_id] += model_score

            if not args.horiz:
                score_count[gpt35_id] += gpt35_score

    res = {}
    for model_id in model_ids:
        if not args.horiz:
            if args.order:
                res[model_id] = {
                    'winning': {
                        'gpt3.5': beat_count[(gpt35_id, model_id)][gpt35_id],
                        'tie': beat_count[(gpt35_id, model_id)]['Tie'],
                        'model': beat_count[(gpt35_id, model_id)][model_id],
                        '%model': beat_count[(gpt35_id, model_id)][model_id] / (n_question - n_skip),
                    },
                    'order': {
                        'gpt3.5': order_count[gpt35_id] / (n_question - n_skip),
                        'model': order_count[model_id] / (n_question - n_skip),
                    },
                    'score': {
                        'gpt3.5': score_count[gpt35_id] / (n_question - n_skip),
                        'model': score_count[model_id] / (n_question - n_skip),
                        '%model': score_count[model_id] / score_count[gpt35_id],
                    },
                }
            else:
                res[model_id] = {
                    'winning': {
                        'gpt3.5': beat_count[(gpt35_id, model_id)][gpt35_id],
                        'tie': beat_count[(gpt35_id, model_id)]['Tie'],
                        'model': beat_count[(gpt35_id, model_id)][model_id],
                        '%model': beat_count[(gpt35_id, model_id)][model_id] / (n_question - n_skip),
                    },
                    'score': {
                        'gpt3.5': score_count[gpt35_id] / (n_question - n_skip),
                        'model': score_count[model_id] / (n_question - n_skip),
                        '%model': score_count[model_id] / score_count[gpt35_id],
                    },
                }
        else:
            if args.order:
                res[model_id] = {
                    'order': order_count[model_id] / (n_question - n_skip),
                    'score': score_count[model_id] / (n_question - n_skip),
                }
            else:
                res[model_id] = {
                    'score': score_count[model_id] / (n_question - n_skip),
                }

    with open(args.output, 'w') as f:
        f.write(json.dumps(res, indent=4) + '\n')
