import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--review')
    parser.add_argument('-o', '--output')
    parser.add_argument('--horiz', action='store_true')
    args = parser.parse_args()

    review = json.load(open(f'{args.review}', 'r'))

    info = []
    if not args.horiz:
        for model_name, value in review.items():
            info.append((model_name, value['order']['model']))

        gpt35_info = ('turbo', value['order']['gpt3.5'])
        info.append(gpt35_info)
    else:
        for model_name, value in review.items():
            info.append((model_name, value['order']))

    ordered_info = sorted(info, key=lambda x: x[1])

    comp_info = []
    cur_idx = 0
    num_eq = 0
    last_order = 0
    for model_name, order in ordered_info:
        if order > last_order:
            cur_idx += num_eq + 1
            comp_info.append((model_name, order, cur_idx))
            num_eq = 0
        else:
            comp_info.append((model_name, order, cur_idx))
            num_eq += 1
        last_order = order

    with open(args.output, 'w') as f:
        f.write('{0:<30}{1:<30}{2:<30}\n'.format('Model', 'Ranking Score', 'Rank'))
        for x in comp_info:
            f.write('{0:<30}{1:<30}{2:<30}'.format(x[0], x[1], x[2]))
            f.write('\n')
