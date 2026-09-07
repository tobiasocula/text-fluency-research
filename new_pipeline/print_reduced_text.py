import json

output_file = "french_texts_random_subset.jsonl"

# Print the first 5 entries
with open(output_file, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 5:
            break
        data = json.loads(line)
        print(f"Entry {i + 1}:")
        print(f"  ID: {data.get('id', 'N/A')}")
        print(f"  Text: {data.get('text', 'N/A')[:100]}...")  # Print first 100 chars
        print(f"  Metadata: {data.get('metadata', {})}")
        print("---")

# Count the total number of lines
with open(output_file, 'r', encoding='utf-8') as f:
    line_count = sum(1 for _ in f)
print(f"\nTotal entries in the subset: {line_count}")