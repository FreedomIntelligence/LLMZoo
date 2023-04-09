import argparse
import json

import backoff
import openai
import ray
import shortuuid

MODEL_ID = 'gpt-3.5-turbo'


@ray.remote(num_cpus=4)
@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--question')
    parser.add_argument('-o', '--output')
    args = parser.parse_args()

    with open(args.question, 'r') as f:
        lines = f.readlines()

    data_list = []
    for line in lines:
        dic = json.loads(line)
        data_list.append(dic)

    ray.init()

    handles = []
    samples = []
    for sample in data_list:
        idx = sample['question_id']
        ques = sample['text'].strip()
        lang = sample['lang']
        category = sample['category']
        message_log = [{"role": "user", "content": ques}]

        sample = {
            'answer_id': shortuuid.uuid(),
            'model_id': MODEL_ID,
            'question_id': idx,
        }
        handles.append(completions_with_backoff.remote(model='gpt-3.5-turbo', messages=message_log))
        samples.append(sample)

    output_list = ray.get(handles)
    with open(args.output, 'w', encoding='utf-8') as f:
        for idx, output in enumerate(output_list):
            text = output['choices'][0]['message']['content']
            samples[idx]['text'] = text
            samples[idx]['metadata'] = {}
            line = json.dumps(samples[idx], ensure_ascii=False)
            f.write(line + '\n')
