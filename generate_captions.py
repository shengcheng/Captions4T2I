import json
from datasets import Dataset
import random

def find_hard_neg(score, negative):
    sorted_score = sorted(score, key=lambda x: score[x])
    if sorted_score[0] == "sum":
        index = sorted_score[1]
    else:
        index = sorted_score[0]
    a, b = index.split('_')
    if isinstance(negative[a], list):
        return negative[a][int(b)]
    else:
        return negative[a]

with open('data/raw.json') as f:
    data = json.load(f)

# NUM_SUBS = 1
# SUBS_SEL = 'FIRST' # 'FIRST' or 'RANDOM' or 'LAST'
# ACC = 0.5

for ACC in [0.25, 0.5, 0.75]:
    for NUM_SUBS in [0,1,2,3,4]:
        for SUBS_SEL in ['FIRST']:
            image_path = []
            captions = []
            correct = []
            for k, v in data.items():
                image_path.append(k)
                c = []
                t = ''
                if ACC >= random.uniform(0, 1):
                    t += v['base']['caption']
                    c.append(1)
                else:
                    t += find_hard_neg(v['base']['score'], v['base']['negative'])
                    c.append(0)
                ks = list(v.keys())
                ks.remove('extended')
                if NUM_SUBS > 0:
                    mask_sort_by_area = sorted(ks, key=lambda x: v[x]['area'], reverse=True)
                    if SUBS_SEL == 'FIRST':
                        sel_mask = mask_sort_by_area[1:1+NUM_SUBS]
                    elif SUBS_SEL == 'LAST':
                        sel_mask = mask_sort_by_area[-NUM_SUBS:]
                    elif SUBS_SEL == 'RANDOM':
                        sel_mask = random.choice(mask_sort_by_area[1:1+NUM_SUBS])
                    else:
                        raise ValueError('SUBS_SEL should be FIRST, LAST or RANDOM')
                    for m in sel_mask:
                        if ACC >= random.uniform(0, 1):
                            t += ' ' + v[m]['caption']
                            c.append(1)
                        else:
                            t += ' ' + find_hard_neg(v[m]['score'], v[m]['negative'])
                            c.append(0)
                captions.append(t)
                correct.append(c)

            ds = Dataset.from_dict({"image": image_path, "text": captions, "correct": correct})
            ds.save_to_disk(f"data/caption_{NUM_SUBS}_{SUBS_SEL}_{ACC}")