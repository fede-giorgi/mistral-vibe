import json
import random
import os

def main():
    sard_file = os.path.join('ai_pipeline', 'chat_dataset.jsonl')
    train_file = os.path.join('ai_pipeline', 'dataset', 'train.jsonl')
    val_file = os.path.join('ai_pipeline', 'dataset', 'val.jsonl')
    test_file = os.path.join('ai_pipeline', 'dataset', 'test.jsonl')

    print("Reading SARD dataset...")
    with open(sard_file, 'r', encoding='utf-8') as f:
        lines = [line for line in f if line.strip()]

    random.seed(42)  # For reproducibility
    random.shuffle(lines)

    total = len(lines)
    train_end = int(total * 0.8)
    val_end = int(total * 0.9)

    train_lines = lines[:train_end]
    val_lines = lines[train_end:val_end]
    test_lines = lines[val_end:]

    print(f"Splitting SARD dataset (Total: {total}): Train {len(train_lines)}, Val {len(val_lines)}, Test {len(test_lines)}")

    def append_to_file(filepath, lines_to_append):
        if not lines_to_append:
            return
        with open(filepath, 'a', encoding='utf-8') as f:
            for line in lines_to_append:
                f.write(line if line.endswith('\n') else line + '\n')

    append_to_file(train_file, train_lines)
    append_to_file(val_file, val_lines)
    append_to_file(test_file, test_lines)

    print("✅ SARD dataset successfully appended to train, val, and test splits.")

if __name__ == "__main__":
    main()
