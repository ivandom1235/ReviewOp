with open('dataset_builder/input/Laptop_train.csv', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if '"' in line:
            print(f"Line {i}: {line[:100]}...")
            if i > 20: break
