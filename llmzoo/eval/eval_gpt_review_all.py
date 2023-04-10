import argparse
import json
import os
import re

import backoff
import numpy as np
import openai
import ray


@ray.remote(num_cpus=4)
@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def get_eval(content: str, max_tokens: int):
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[{
            'role': 'system',
            'content': 'You are a helpful and precise assistant for checking the quality of the answer.'
        }, {
            'role': 'user',
            'content': content,
        }],
        temperature=0,
        # max_tokens=max_tokens,
    )
    return response['choices'][0]['message']['content']


def parse_score_cot(review):
    try:
        if re.match(r'^[ \d\.]+$', review.strip().split('\n')[-1]):
            score_list = review.strip().split('\n')[-1]
            score_list = score_list.replace(',', ' ')
            sl = score_list.split(' ')
        elif re.search(r'([ \d\.]+)$', review.strip()):
            score_list = re.search(r'([ \d\.]+)$', review).group(1).strip(' .')
            sl = score_list.split(' ')
        else:
            sl = []
            for i in range(n_ans):
                sl.append(re.search(r'Assistant %d: ([0-9\.]+)' % (i + 1), review).group(1))
        if len(sl) == n_ans:
            return [float(s) for s in sl]
        else:
            return [-1] * n_ans
    except Exception as e:
        # print(e)
        # print('error', review)
        return [-1] * n_ans


def parse_order_cot(review):
    review = re.sub(r'>=', '>', review)
    review = re.sub(r'>>', '>', review)
    try:
        ls = re.findall(r'(Assistant \d( [>=] Assistant \d)+)', review.strip())
        order_texts = [x[0] for x in ls]
        idxs = np.where(np.array([len(re.findall(r'Assistant', text)) == n_ans for text in order_texts]))[0]
        if idxs.shape[0] == 0:
            return [-1] * n_ans
        order_text = order_texts[idxs[0]]

        ordered_assist = [int(x) for x in re.findall(r'\d', order_text)]
        ordered_comp = re.findall(r'[>=]', order_text)

        order = [0] * n_ans
        cur_order = 1
        num_eq = 0
        order[ordered_assist[0] - 1] = cur_order
        for comp, assist in zip(ordered_comp, ordered_assist[1:]):
            if comp == '>':
                cur_order += num_eq + 1
                order[assist - 1] = cur_order
                num_eq = 0
            else:
                order[assist - 1] = cur_order
                num_eq += 1
        return order

    except Exception as e:
        # print(e)
        # print('error', review)
        return [-1] * n_ans


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


num2str = {2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine'}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChatGPT-based QA evaluation.')
    parser.add_argument('-q', '--question')
    parser.add_argument('-a', '--answer-list', nargs='+', default=[])
    parser.add_argument('-r', '--rule')
    parser.add_argument('-o', '--output')
    parser.add_argument('--max-tokens', type=int, default=1024, help='maximum number of tokens produced in the output')
    parser.add_argument('--dimension', choices=['general', 'relevance', 'diversity', 'coherence', 'immersion'])
    parser.add_argument('--order', action='store_true')
    args = parser.parse_args()

    ray.init()

    questions = read_jsonl(args.question, key='question_id')
    ans_dict_list = [read_jsonl(answer_file, key='question_id') for answer_file in args.answer_list]
    rule = json.load(open(os.path.expanduser(args.rule), 'r'))[args.dimension]
    assert all([len(set(questions.keys()) - set(ans_dict.keys())) == 0 for ans_dict in ans_dict_list])

    review_file = open(f'{args.output}', 'w')

    global n_ans
    n_ans = len(ans_dict_list)
    n_ans_str = num2str[n_ans]

    js_list = []
    handles = []
    idx = 0
    for ques_id in questions.keys():
        # if idx == 5:
        #     break

        ques_dict = questions[ques_id]
        ques = ques_dict['text']
        lang = ques_dict.get('lang', 'en')
        category = ques_dict['category']

        ans_list = [ans_dict[ques_id]['text'] for ans_dict in ans_dict_list]

        prompt = rule['prompt'].format(num=n_ans, num_str=n_ans_str)
        assert (args.order and 'order' in prompt) or (not args.order and 'score' in prompt)
        role = rule['role']
        content = f'[Question]\n{ques}\n\n'
        content += '\n\n'.join(
            [f'[{role} {i}]\n{ans}\n\n[End of {role} {i}]' for i, ans in enumerate(ans_list, start=1)])
        content += '\n\n' + f'[System]\n{prompt}\n\n'

        js_list.append({
            # 'review_id': idx+1,
            'reviewer_id': 'gpt-3.5-turbo',
            'lang': lang,
            'question_id': ques_id,
            'answer_ids': [ans_dict[ques_id]['answer_id'] for ans_dict in ans_dict_list],
            'category': category,
            'metadata': {
                'question': ques,
                'answers': ans_list,
                'model_ids': [ans_dict[ques_id]['model_id'] for ans_dict in ans_dict_list]
            }})
        idx += 1
        handles.append(get_eval.remote(content, args.max_tokens))

    n_errors = 0
    reviews = ray.get(handles)
    for idx, review in enumerate(reviews):
        js_list[idx]['text'] = review
        if args.order:
            orders = parse_order_cot(review)
            js_list[idx]['order'] = orders
            if -1 in orders:
                n_errors += 1
        else:
            scores = parse_score_cot(review)
            js_list[idx]['score'] = scores
            if -1 in scores:
                n_errors += 1
        review_file.write(json.dumps(js_list[idx], ensure_ascii=False) + '\n')
    review_file.close()
    print(f'There are {n_errors} reviews failing to decode.')
