#!/usr/bin/env python3
"""
High-Entropy File Generator
Generates files with maximum entropy that are resistant to compression.
Each byte is cryptographically random and independent of all others.
"""

import gzip
import lzma
import os
import shutil
import subprocess
import sys
import argparse
from pathlib import Path
from typing import Callable, Optional


class EntropyFileGenerator:
    """Generates files with maximum entropy for compression resistance testing."""
    
    def __init__(self, file_size: int, row_width: Optional[int] = None):
        """
        Initialize the entropy file generator.
        
        Args:
            file_size: Total size of the file to generate in bytes
            row_width: Optional width of each row in bytes (for structured output)
        """
        self.file_size = file_size
        self.row_width = row_width if row_width else file_size
        
        if self.row_width > self.file_size:
            self.row_width = self.file_size
    
    def generate(self, output_path: str, chunk_size: int = 1024 * 1024) -> None:
        """
        Generate a high-entropy file.
        
        Uses os.urandom() which reads from /dev/urandom on macOS/Linux,
        providing cryptographically strong random bytes. Each byte is
        completely independent and unpredictable.
        
        Args:
            output_path: Path where the file will be created
            chunk_size: Size of chunks to write at once (default 1MB)
        """
        bytes_written = 0
        
        try:
            with open(output_path, 'wb') as f:
                while bytes_written < self.file_size:
                    # Calculate how many bytes to generate in this iteration
                    remaining = self.file_size - bytes_written
                    current_chunk_size = min(chunk_size, remaining)
                    
                    # Generate cryptographically random bytes
                    # os.urandom uses /dev/urandom on macOS, which provides
                    # non-blocking, cryptographically secure random data
                    random_bytes = os.urandom(current_chunk_size)
                    
                    # Write the random bytes
                    f.write(random_bytes)
                    bytes_written += current_chunk_size
                    
                    # Progress indicator for large files
                    if self.file_size > 10 * 1024 * 1024:  # Show progress for files > 10MB
                        progress = (bytes_written / self.file_size) * 100
                        print(f"\rProgress: {progress:.1f}% ({bytes_written:,} / {self.file_size:,} bytes)", 
                              end='', file=sys.stderr)
            
            if self.file_size > 10 * 1024 * 1024:
                print(file=sys.stderr)  # New line after progress
                
            print(f"Successfully generated {bytes_written:,} bytes of high-entropy data")
            print(f"Output file: {output_path}")
            
        except IOError as e:
            print(f"Error writing file: {e}", file=sys.stderr)
            sys.exit(1)
    
    def analyze_entropy(self, data: bytes, sample_size: int = 10000) -> dict:
        """
        Analyze a sample of the generated data to verify high entropy.
        
        Args:
            data: Byte data to analyze
            sample_size: Number of bytes to analyze
            
        Returns:
            Dictionary with entropy statistics
        """
        sample = data[:min(sample_size, len(data))]
        
        # Count byte frequency
        freq = [0] * 256
        for byte in sample:
            freq[byte] += 1
        
        # Calculate Shannon entropy
        import math
        entropy = 0.0
        for count in freq:
            if count > 0:
                probability = count / len(sample)
                entropy -= probability * math.log2(probability)
        
        # Count unique bytes
        unique_bytes = sum(1 for count in freq if count > 0)
        
        return {
            'entropy': entropy,
            'max_entropy': 8.0,
            'unique_bytes': unique_bytes,
            'sample_size': len(sample)
        }


# --- Compression algorithms (max levels); original file is never modified ---

def _compress_gzip(path: Path, out_path: Path, chunk_size: int) -> None:
    """Gzip at maximum level (9)."""
    with open(path, 'rb') as f_in:
        with gzip.open(out_path, 'wb', compresslevel=9) as f_out:
            shutil.copyfileobj(f_in, f_out, length=chunk_size)


def _compress_xz(path: Path, out_path: Path, chunk_size: int) -> None:
    """XZ/LZMA at maximum preset (9)."""
    with open(path, 'rb') as f_in:
        with lzma.open(out_path, 'wb', preset=9) as f_out:
            shutil.copyfileobj(f_in, f_out, length=chunk_size)


def _compress_zstd(path: Path, out_path: Path, _chunk_size: int) -> None:
    """Zstandard via CLI at max level (19); preserves original."""
    try:
        subprocess.run(
            ['zstd', '-19', '-f', '-q', '-o', str(out_path), str(path)],
            check=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise RuntimeError("zstd CLI not available or failed (install zstd)") from e


def _compress_brotli(path: Path, out_path: Path, _chunk_size: int) -> None:
    """Brotli via CLI at max quality (11); preserves original."""
    try:
        subprocess.run(
            ['brotli', '-q', '11', '-f', '-o', str(out_path), str(path)],
            check=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise RuntimeError("brotli CLI not available or failed (install brotli)") from e


def _compress_paq(path: Path, out_path: Path, _chunk_size: int) -> None:
    """PAQ8 via CLI if available (e.g. paq8px) at max level (-8). Output varies by variant."""
    before = set(path.parent.iterdir())
    for cmd in ('paq8px', 'paq8px_v', 'paq8l'):
        try:
            subprocess.run(
                [cmd, '-8', str(path)],
                capture_output=True,
                text=True,
                timeout=600,
                cwd=str(path.parent),
            )
            after = set(path.parent.iterdir())
            created = after - before - {path}
            for c in created:
                if c.is_file():
                    shutil.move(str(c), str(out_path))
                    return
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            continue
    raise RuntimeError("No PAQ8 CLI found (paq8px/paq8l not in PATH)")


COMPRESSORS: dict[str, tuple[str, Callable[[Path, Path, int], None]]] = {
    'gzip':   ('.gz',   _compress_gzip),
    'xz':     ('.xz',   _compress_xz),
    'zstd':   ('.zst',  _compress_zstd),
    'brotli': ('.br',   _compress_brotli),
    'paq':    ('.paq8', _compress_paq),
}


def compress_and_compare(
    uncompressed_path: str,
    chunk_size: int = 1024 * 1024,
    algorithms: Optional[list[str]] = None,
) -> None:
    """
    Compress the file with selected lossless algorithms at max levels and compare.
    Original file is preserved; each algorithm writes to a separate file (e.g. .gz, .xz, .zst).
    Reports: original size, compressed size, ratio (compressed/original).
    """
    path = Path(uncompressed_path)
    uncompressed_size = path.stat().st_size
    if uncompressed_size == 0:
        print("\nNo compression (file is empty).")
        return

    to_run = algorithms if algorithms else ['gzip', 'xz']  # stdlib-only default
    results: list[tuple[str, Path, int, float]] = []

    print()
    print("Compression comparison (entropy vs lossless at max level):")
    print(f"  Original size: {uncompressed_size:,} bytes ({format_size(uncompressed_size)})")
    print()

    for name in to_run:
        if name not in COMPRESSORS:
            print(f"  [{name}] unknown algorithm (skip)")
            continue
        ext, compress_fn = COMPRESSORS[name]
        out_path = path.with_suffix(path.suffix + ext)
        try:
            compress_fn(path, out_path, chunk_size)
            compressed_size = out_path.stat().st_size
            ratio = compressed_size / uncompressed_size
            results.append((name, out_path, compressed_size, ratio))
            print(f"  {name:8}  {compressed_size:>12,} bytes  ratio {ratio:.4f}  -> {out_path}")
        except Exception as e:
            print(f"  {name:8}  skipped: {e}")

    if not results:
        print("  No algorithms succeeded.")
        return

    best = min(results, key=lambda r: r[2])
    print()
    if best[2] >= uncompressed_size:
        print("  Entropy wins: best compressed size >= original (compression useless)")
    elif best[3] > 0.99:
        print("  Entropy wins: negligible compression (high-entropy data)")
    else:
        print(f"  Best: {best[0]}  (compression saved {100 * (1 - best[3]):.2f}%)")


def format_size(size_bytes: int) -> str:
    """Format byte size in human-readable form."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def parse_size(size_str: str) -> int:
    """
    Parse size string with optional unit suffix (K, M, G).
    
    Examples:
        "1024" -> 1024 bytes
        "10K" -> 10240 bytes
        "5M" -> 5242880 bytes
        "1G" -> 1073741824 bytes
    """
    size_str = size_str.strip().upper()
    
    multipliers = {
        'K': 1024,
        'M': 1024 * 1024,
        'G': 1024 * 1024 * 1024,
        'T': 1024 * 1024 * 1024 * 1024
    }
    
    if size_str[-1] in multipliers:
        try:
            number = float(size_str[:-1])
            return int(number * multipliers[size_str[-1]])
        except ValueError:
            raise ValueError(f"Invalid size format: {size_str}")
    else:
        try:
            return int(size_str)
        except ValueError:
            raise ValueError(f"Invalid size format: {size_str}")


def main():
    """Main entry point for the entropy file generator."""
    parser = argparse.ArgumentParser(
        description='Generate high-entropy files resistant to compression',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s output.bin -s 1M              # Generate 1MB file
  %(prog)s output.bin -s 1M -z           # Generate 1MB, compress with gzip+xz at max level, compare
  %(prog)s output.bin -s 1M -z -A gzip,xz,zstd,brotli  # Try gzip, xz, zstd, brotli (zstd/brotli need CLI)
  %(prog)s output.bin -s 100M -w 1024    # Generate 100MB file with 1024-byte rows
  %(prog)s test.dat -s 1G -a             # Generate 1GB file and analyze entropy
  %(prog)s data.bin -s 10485760          # Generate 10MB file (size in bytes)

Size suffixes: K (kilobytes), M (megabytes), G (gigabytes), T (terabytes)

Notes:
  - Uses cryptographically secure random number generator (os.urandom)
  - Each byte is completely independent and unpredictable
  - Ideal for testing compression algorithms
  - On macOS, uses /dev/urandom as the entropy source
        """)
    
    parser.add_argument('output', 
                       help='Output file path')
    parser.add_argument('-s', '--size', 
                       required=True,
                       help='File size (e.g., 1M, 100K, 1G, or bytes)')
    parser.add_argument('-w', '--width', 
                       type=int,
                       help='Row width in bytes (optional, for structured output)')
    parser.add_argument('-a', '--analyze', 
                       action='store_true',
                       help='Analyze entropy of generated data')
    parser.add_argument('-z', '--compress',
                       action='store_true',
                       help='Compress output with selected algorithms at max level and compare sizes')
    parser.add_argument('-A', '--algorithm',
                       metavar='ALGOS',
                       default=None,
                       help='Comma-separated algorithms for -z: gzip, xz, zstd, brotli, paq (default: gzip,xz; zstd/brotli/paq need CLI)')
    parser.add_argument('-c', '--chunk-size',
                       default='1M',
                       help='Write chunk size (default: 1M)')
    
    args = parser.parse_args()
    
    try:
        file_size = parse_size(args.size)
        chunk_size = parse_size(args.chunk_size)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    if file_size <= 0:
        print("Error: File size must be positive", file=sys.stderr)
        sys.exit(1)
    
    if args.width and args.width <= 0:
        print("Error: Row width must be positive", file=sys.stderr)
        sys.exit(1)
    
    print(f"Generating high-entropy file...")
    print(f"Size: {format_size(file_size)} ({file_size:,} bytes)")
    if args.width:
        print(f"Row width: {args.width} bytes")
    print(f"Output: {args.output}")
    print()
    
    # Generate the file
    generator = EntropyFileGenerator(file_size, args.width)
    generator.generate(args.output, chunk_size)

    # Optionally compress and compare (verify entropy beats compression)
    if args.compress:
        algos = None
        if args.algorithm:
            algos = [a.strip().lower() for a in args.algorithm.split(',') if a.strip()]
        compress_and_compare(args.output, chunk_size, algorithms=algos)
    
    # Optionally analyze entropy
    if args.analyze:
        print("\nAnalyzing entropy...")
        with open(args.output, 'rb') as f:
            sample_data = f.read(min(100000, file_size))
        
        stats = generator.analyze_entropy(sample_data)
        print(f"\nEntropy Analysis:")
        print(f"  Shannon Entropy: {stats['entropy']:.6f} bits/byte")
        print(f"  Maximum Entropy: {stats['max_entropy']:.6f} bits/byte")
        print(f"  Entropy Ratio:   {(stats['entropy']/stats['max_entropy'])*100:.2f}%")
        print(f"  Unique Bytes:    {stats['unique_bytes']}/256")
        print(f"  Sample Size:     {stats['sample_size']:,} bytes")
        
        if stats['entropy'] > 7.99:
            print("\n✓ Excellent: File has maximum entropy and is highly compression-resistant")
        elif stats['entropy'] > 7.9:
            print("\n✓ Good: File has very high entropy")
        else:
            print("\n⚠ Warning: Entropy is lower than expected")


if __name__ == '__main__':
    main()