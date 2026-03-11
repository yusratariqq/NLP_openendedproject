import json
import pickle
import os
from collections import Counter

def convert_label(raw_label: str) -> str:
    label = raw_label.lower().strip()
    if label == 'true':
        return 'SUPPORTS'
    elif label == 'false':
        return 'REFUTES'
    else:
        return 'NEI'

def build_jsonl(examples: list) -> list:
    output = []
    for ex in examples:
        evidence_parts = []
        for i in range(1, 6):
            ev = ex.get(f'evidence_{i}', '')
            if ev and ev != '<DUMMY_EVIDENCE>' and ev is not None:
                evidence_parts.append(ev.strip())

        evidence = ' '.join(evidence_parts[:2])

        output.append({
            'claim':    ex['claim'],
            'evidence': evidence,
            'label':    convert_label(ex['label']),
            'language': ex['language']
        })
    return output

# Load XFACT
with open('/content/data/cache/xfact_data.pkl', 'rb') as f:
    xfact_data = pickle.load(f)

os.makedirs('data/final', exist_ok=True)

split_map = {'train': 'train', 'dev': 'val', 'test': 'test'}

for xfact_split, file_split in split_map.items():
    examples = xfact_data.get(xfact_split, [])
    converted = build_jsonl(examples)

    path = f'data/final/{file_split}.jsonl'
    with open(path, 'w', encoding='utf-8') as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    labels = Counter(ex['label'] for ex in converted)
    langs  = Counter(ex['language'] for ex in converted)
    print(f"{file_split}: {len(converted)} examples | labels: {dict(labels)} | langs: {dict(langs)}")

print("\n✅ data/final/ ready for training")
