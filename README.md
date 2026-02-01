# uncomp

<video src="asset/uncompresable.mp4" controls width="100%"></video>

**uncomp** generates high-entropy files that resist lossless compression. It demonstrates that under maximum entropy, compression algorithms cannot reduce (and may increase) file size.

## The idea

We all use compression (gzip, zip, xz, zstd, brotli). The idea is simple: the algorithm finds repeated patterns and encodes them more compactly—e.g. `AAAA` → `4A`.

What defeats every compression algorithm? A **high-entropy file**: one where every byte is random and independent. There are no patterns to exploit.

How do the algorithms behave? Literally, they can’t help—the file size may **increase** because of metadata and framing. Under total randomness and chaos, compression is useless.

## The experiment

Generate a 1 MB high-entropy file and compress it with several algorithms at maximum level:

```bash
python3 main.py output.bin -s 1M -z -A gzip,xz,zstd,brotli
```

**Example results:**

| Original | gzip   | xz     | zstd   | brotli |
|----------|--------|--------|--------|--------|
| 1,048,576 B | 1,048,925 B | 1,048,688 B | 1,048,613 B | 1,048,584 B |

**Entropy wins:** compressed size ≥ original. The algorithms add overhead; they don’t shrink the data.

## The analogy (from the podcast)

Inspired by the Episode from The Rest Is Science *Searching For Meaning In Randomness*:

- **Repetitive pattern:** “I rolled the dice 10 times and got 6 every time.” One pattern, easy to compress.
- **Fully random:** “I rolled 10 times: once 3, once 1, once 5…” No pattern; each outcome is independent. That’s maximum entropy—and that’s exactly what we feed to gzip, xz, zstd, and brotli. They can’t reduce the size because there’s nothing to compress.

**Episode:** [https://youtu.be/tiXIOpq_tQ0](https://youtu.be/tiXIOpq_tQ0)

## Usage

```bash
# Generate 1 MB high-entropy file
python3 main.py output.bin -s 1M

# Generate, then compress with gzip + xz (default) and compare sizes
python3 main.py output.bin -s 1M -z

# Try all algorithms (zstd/brotli require CLI tools)
python3 main.py output.bin -s 1M -z -A gzip,xz,zstd,brotli

# Analyze entropy of generated data
python3 main.py output.bin -s 1M -a
```

**Options:** `-s` size (e.g. `1M`, `100K`), `-w` row width, `-z` compress and compare, `-A` algorithms (gzip,xz,zstd,brotli,paq), `-a` analyze entropy.

## How it works

The tool uses the OS CSPRNG (`os.urandom()`, e.g. `/dev/urandom` on macOS/Linux) so each byte is cryptographically random and independent. No patterns, no structure—maximum entropy. Lossless compressors have nothing to exploit, so the “compressed” output is at least as large as the original (often slightly larger due to headers and metadata).
