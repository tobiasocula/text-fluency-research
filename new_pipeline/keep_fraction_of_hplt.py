import zstandard as zstd
from pathlib import Path
import random

input_file = Path.cwd() / "hun_Latn" / "10_1.jsonl.zst"
output_file = "hun_texts_subset.jsonl"
# sample_ratio = 0.01  # 1% of the data

max_lines = 100_000

with open(input_file, 'rb') as fh:
    dctx = zstd.ZstdDecompressor()
    with dctx.stream_reader(fh) as reader:
        buffer = ""
        line_count = 0
        with open(output_file, 'w', encoding='utf-8') as out_fh:
            while True:
                chunk = reader.read(8192)  # Read 8KB at a time
                if not chunk:
                    break  # End of file

                # Decode the chunk and handle incomplete UTF-8 sequences
                try:
                    decoded_chunk = chunk.decode('utf-8')
                except UnicodeDecodeError:
                    # If decoding fails, try to find the last valid UTF-8 character
                    for i in range(len(chunk), 0, -1):
                        try:
                            decoded_chunk = chunk[:i].decode('utf-8')
                            buffer += decoded_chunk
                            # Keep the remaining bytes in the buffer for the next chunk
                            chunk = chunk[i:]
                            break
                        except UnicodeDecodeError:
                            continue
                    else:
                        # If no valid UTF-8 sequence is found, skip the chunk
                        continue

                buffer += decoded_chunk
                lines = buffer.splitlines(keepends=True)
                buffer = lines.pop() if lines else ""  # Keep the last incomplete line

                if line_count > max_lines:
                    break

                for line in lines:
                    out_fh.write(line)
                    line_count += 1

print(f"Random sample saved to {output_file}")