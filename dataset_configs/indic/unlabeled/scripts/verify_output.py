#!/usr/bin/env python3
"""
Verification Script for Indic Audio Processing Output

This script verifies the quality and statistics of processed Indic audio manifests.

Usage:
    python verify_output.py --manifest /path/to/manifest.json
    python verify_output.py --workspace /data/hindi_audio --language hi
"""

import argparse
import json
import os
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Any


def load_manifest(manifest_path: str) -> List[Dict[str, Any]]:
    """Load manifest file and return list of entries."""
    data = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def compute_statistics(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute statistics from manifest data."""
    stats = {
        'total_samples': len(data),
        'total_duration': 0.0,
        'avg_duration': 0.0,
        'min_duration': float('inf'),
        'max_duration': 0.0,
        'total_words': 0,
        'avg_words': 0.0,
        'empty_text': 0,
        'missing_fields': defaultdict(int),
    }
    
    duration_buckets = defaultdict(int)
    word_count_buckets = defaultdict(int)
    
    for entry in data:
        # Duration statistics
        if 'duration' in entry:
            duration = entry['duration']
            stats['total_duration'] += duration
            stats['min_duration'] = min(stats['min_duration'], duration)
            stats['max_duration'] = max(stats['max_duration'], duration)
            
            # Bucket by duration
            bucket = int(duration)
            duration_buckets[bucket] += 1
        else:
            stats['missing_fields']['duration'] += 1
        
        # Text statistics
        if 'text' in entry:
            text = entry['text'].strip()
            if not text:
                stats['empty_text'] += 1
        else:
            stats['missing_fields']['text'] += 1
        
        # Word count statistics
        if 'num_words' in entry:
            num_words = entry['num_words']
            stats['total_words'] += num_words
            
            # Bucket by word count
            if num_words < 5:
                word_count_buckets['0-4'] += 1
            elif num_words < 10:
                word_count_buckets['5-9'] += 1
            elif num_words < 20:
                word_count_buckets['10-19'] += 1
            elif num_words < 50:
                word_count_buckets['20-49'] += 1
            else:
                word_count_buckets['50+'] += 1
        
        # Check for audio file
        if 'audio_filepath' in entry:
            if not os.path.exists(entry['audio_filepath']):
                stats['missing_fields']['audio_file'] += 1
        else:
            stats['missing_fields']['audio_filepath'] += 1
    
    # Compute averages
    if stats['total_samples'] > 0:
        stats['avg_duration'] = stats['total_duration'] / stats['total_samples']
        stats['avg_words'] = stats['total_words'] / stats['total_samples']
    
    if stats['min_duration'] == float('inf'):
        stats['min_duration'] = 0.0
    
    stats['duration_buckets'] = dict(sorted(duration_buckets.items()))
    stats['word_count_buckets'] = dict(word_count_buckets)
    
    return stats


def check_indic_script(text: str, language_code: str) -> bool:
    """Check if text contains appropriate Indic script characters."""
    # Unicode ranges for Indic scripts
    indic_ranges = {
        'hi': (0x0900, 0x097F),  # Devanagari
        'mr': (0x0900, 0x097F),  # Devanagari
        'te': (0x0C00, 0x0C7F),  # Telugu
        'ta': (0x0B80, 0x0BFF),  # Tamil
        'bn': (0x0980, 0x09FF),  # Bengali
        'as': (0x0980, 0x09FF),  # Bengali
        'ml': (0x0D00, 0x0D7F),  # Malayalam
        'kn': (0x0C80, 0x0CFF),  # Kannada
        'gu': (0x0A80, 0x0AFF),  # Gujarati
        'pa': (0x0A00, 0x0A7F),  # Gurmukhi
        'or': (0x0B00, 0x0B7F),  # Odia
        'en': (0x0041, 0x007A),  # Latin (English)
    }
    
    if language_code not in indic_ranges:
        return True  # Unknown language, skip check
    
    start, end = indic_ranges[language_code]
    
    # Count characters in the appropriate script
    script_chars = sum(1 for char in text if start <= ord(char) <= end)
    total_chars = len([c for c in text if c.isalpha()])
    
    if total_chars == 0:
        return False
    
    # At least 50% should be in the expected script
    return (script_chars / total_chars) >= 0.5


def verify_script_consistency(data: List[Dict[str, Any]], language_code: str) -> Dict[str, int]:
    """Verify that text contains appropriate script for the language."""
    results = {
        'total_checked': 0,
        'valid_script': 0,
        'invalid_script': 0,
    }
    
    for entry in data:
        if 'text' in entry and entry['text'].strip():
            results['total_checked'] += 1
            if check_indic_script(entry['text'], language_code):
                results['valid_script'] += 1
            else:
                results['invalid_script'] += 1
    
    return results


def print_statistics(stats: Dict[str, Any], language_name: str = None):
    """Print formatted statistics."""
    print("\n" + "=" * 70)
    if language_name:
        print(f"Manifest Verification Report - {language_name}")
    else:
        print("Manifest Verification Report")
    print("=" * 70)
    
    print(f"\n📊 Overall Statistics:")
    print(f"  Total Samples: {stats['total_samples']:,}")
    print(f"  Total Duration: {stats['total_duration'] / 3600:.2f} hours")
    print(f"  Average Duration: {stats['avg_duration']:.2f} seconds")
    print(f"  Min Duration: {stats['min_duration']:.2f} seconds")
    print(f"  Max Duration: {stats['max_duration']:.2f} seconds")
    
    print(f"\n📝 Text Statistics:")
    print(f"  Total Words: {stats['total_words']:,}")
    print(f"  Average Words per Sample: {stats['avg_words']:.2f}")
    print(f"  Empty Transcriptions: {stats['empty_text']}")
    
    if stats['word_count_buckets']:
        print(f"\n📈 Word Count Distribution:")
        for bucket, count in sorted(stats['word_count_buckets'].items()):
            percentage = (count / stats['total_samples']) * 100
            print(f"  {bucket:>6} words: {count:>6} ({percentage:>5.1f}%)")
    
    if stats['missing_fields']:
        print(f"\n⚠️  Missing/Invalid Fields:")
        for field, count in stats['missing_fields'].items():
            print(f"  {field}: {count}")
    
    if stats.get('script_verification'):
        sv = stats['script_verification']
        print(f"\n🔤 Script Verification:")
        print(f"  Total Checked: {sv['total_checked']}")
        print(f"  Valid Script: {sv['valid_script']}")
        print(f"  Invalid Script: {sv['invalid_script']}")
        if sv['total_checked'] > 0:
            valid_pct = (sv['valid_script'] / sv['total_checked']) * 100
            print(f"  Validity: {valid_pct:.1f}%")


def print_sample_entries(data: List[Dict[str, Any]], num_samples: int = 5):
    """Print sample entries from manifest."""
    print(f"\n📄 Sample Entries (showing first {num_samples}):")
    print("-" * 70)
    
    for i, entry in enumerate(data[:num_samples], 1):
        print(f"\nSample {i}:")
        print(f"  Audio: {Path(entry.get('audio_filepath', 'N/A')).name}")
        print(f"  Duration: {entry.get('duration', 0):.2f}s")
        print(f"  Text: {entry.get('text', 'N/A')[:80]}...")
        if 'num_words' in entry:
            print(f"  Words: {entry['num_words']}")


def main():
    parser = argparse.ArgumentParser(
        description='Verify Indic audio processing output manifest'
    )
    parser.add_argument(
        '--manifest',
        type=str,
        help='Path to manifest file'
    )
    parser.add_argument(
        '--workspace',
        type=str,
        help='Workspace directory (alternative to --manifest)'
    )
    parser.add_argument(
        '--language',
        type=str,
        help='Language code (required with --workspace)'
    )
    parser.add_argument(
        '--show-samples',
        type=int,
        default=5,
        help='Number of sample entries to show (default: 5)'
    )
    parser.add_argument(
        '--verify-script',
        action='store_true',
        help='Verify that text uses appropriate Indic script'
    )
    
    args = parser.parse_args()
    
    # Determine manifest path
    if args.manifest:
        manifest_path = args.manifest
        language_code = args.language
    elif args.workspace and args.language:
        manifest_path = os.path.join(
            args.workspace,
            'manifests',
            f'{args.language}_final_manifest.json'
        )
        language_code = args.language
    else:
        parser.error('Either --manifest or both --workspace and --language are required')
    
    # Language names
    language_names = {
        'hi': 'Hindi', 'te': 'Telugu', 'ta': 'Tamil', 'bn': 'Bengali',
        'ml': 'Malayalam', 'kn': 'Kannada', 'mr': 'Marathi', 'gu': 'Gujarati',
        'pa': 'Punjabi', 'or': 'Odia', 'as': 'Assamese', 'en': 'English'
    }
    
    language_name = language_names.get(language_code, language_code) if language_code else None
    
    # Check if file exists
    if not os.path.exists(manifest_path):
        print(f"❌ Error: Manifest file not found: {manifest_path}")
        return 1
    
    print(f"📂 Loading manifest: {manifest_path}")
    
    # Load and analyze
    try:
        data = load_manifest(manifest_path)
        stats = compute_statistics(data)
        
        # Script verification if requested
        if args.verify_script and language_code:
            stats['script_verification'] = verify_script_consistency(data, language_code)
        
        # Print results
        print_statistics(stats, language_name)
        
        if args.show_samples > 0:
            print_sample_entries(data, args.show_samples)
        
        # Quality checks
        print("\n" + "=" * 70)
        print("✓ Verification Complete")
        
        issues = []
        if stats['empty_text'] > 0:
            issues.append(f"Found {stats['empty_text']} empty transcriptions")
        if stats['missing_fields']:
            issues.append(f"Missing fields detected")
        if stats.get('script_verification', {}).get('invalid_script', 0) > 0:
            issues.append(f"Found text with incorrect script")
        
        if issues:
            print("\n⚠️  Issues Found:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("\n✅ No major issues detected")
        
        print("=" * 70 + "\n")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error processing manifest: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())

