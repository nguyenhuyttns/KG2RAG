import json

def create_subset():
    input_path = '../../data/hotpotqa/hotpot_dev_distractor_v1.json'
    output_path = '../../data/hotpotqa/hotpot_dev_distractor_test.json'
    
    print(f"Reading dataset from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    subset = data[:50]
    print(f"Extracted {len(subset)} samples. Saving to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(subset, f, indent=2, ensure_ascii=False)
    print("Done!")

if __name__ == '__main__':
    create_subset()
