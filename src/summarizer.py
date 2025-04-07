#!/usr/bin/env python
"""
summarizer.py
───────────────────────────────────────────────────────────────────────────────
Process TTS evaluation results from VERSA, which are in a line-by-line JSON format
where each line contains a single JSON object.

This script:
1. Reads the line-by-line format directly (already JSONL-like)
2. Calculates summary statistics from dependent and independent metrics
3. Generates a comprehensive summary report

Usage:
    python summarizer.py --independent path/to/independent.json --dependent path/to/dependent.json --output path/to/output_directory

Example:
    python summarizer.py --independent evaluation_results/independent.json --dependent evaluation_results/dependent.json --output evaluation_results
"""

import argparse
import json
import logging
import statistics
from pathlib import Path
from typing import Dict, List, Any, Union


def read_line_by_line_json(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Read data from a file where each line is a JSON object.
    
    Args:
        file_path: Path to file with JSON objects on each line
    
    Returns:
        List of dictionaries containing the parsed JSON data
    """
    data = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        # Replace single quotes with double quotes for valid JSON
                        line = line.replace("'", '"')
                        # Handle 'inf' value which is not valid JSON
                        line = line.replace('inf', '"inf"')
                        item = json.loads(line)
                        data.append(item)
                    except json.JSONDecodeError as e:
                        logging.warning(f"Failed to parse line: {line}, Error: {e}")
        return data
    except FileNotFoundError:
        logging.warning(f"File not found: {file_path}")
        return []


def write_jsonl(data: List[Dict[str, Any]], output_file: Union[str, Path]) -> None:
    """
    Write data to a JSONL file (one JSON object per line).
    
    Args:
        data: List of dictionaries to write
        output_file: Path to output JSONL file
    """
    with open(output_file, 'w') as f:
        for item in data:
            # Convert 'inf' string back to appropriate value
            for k, v in item.items():
                if v == "inf":
                    item[k] = float('inf')
            f.write(json.dumps(item) + '\n')
    
    logging.info(f"Wrote JSONL data to: {output_file}")


def calculate_metrics_summary(independent_data: List[Dict[str, Any]], 
                             dependent_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate summary statistics from dependent and independent metrics.
    
    Args:
        independent_data: List of dictionaries containing independent metrics
        dependent_data: List of dictionaries containing dependent metrics
    
    Returns:
        Dictionary containing summary statistics
    """
    summary = {}
    
    # Process independent metrics (e.g., UTMOS)
    if independent_data:
        utmos_values = [item.get('utmos') for item in independent_data if 'utmos' in item]
        if utmos_values:
            summary['utmos_mean'] = statistics.mean(utmos_values)
            summary['utmos_median'] = statistics.median(utmos_values)
            summary['utmos_min'] = min(utmos_values)
            summary['utmos_max'] = max(utmos_values)
            summary['utmos_std'] = statistics.stdev(utmos_values) if len(utmos_values) > 1 else 0
    
    # Process dependent metrics (e.g., PESQ, STOI, etc.)
    if dependent_data:
        # Create a summary for each metric found
        metrics = set()
        for item in dependent_data:
            metrics.update(key for key in item.keys() if key != 'key')
        
        for metric in metrics:
            values = [item.get(metric) for item in dependent_data if metric in item]
            # Filter out None, inf, and -inf values
            values = [v for v in values if v is not None and 
                     (isinstance(v, (int, float)) and 
                      v != float('inf') and v != float('-inf') and
                      v != "inf")]
            
            if values:
                summary[f'{metric}_mean'] = statistics.mean(values)
                summary[f'{metric}_median'] = statistics.median(values)
                summary[f'{metric}_min'] = min(values)
                summary[f'{metric}_max'] = max(values)
                summary[f'{metric}_std'] = statistics.stdev(values) if len(values) > 1 else 0
    
    # Calculate combined quality score (example: weighted average of UTMOS and PESQ)
    if 'utmos_mean' in summary and 'pesq_mean' in summary:
        # Normalize PESQ to similar scale as UTMOS (PESQ is typically 1-4.5, UTMOS is often 1-5)
        normalized_pesq = (summary['pesq_mean'] - 1) / 3.5 * 4 + 1
        summary['combined_quality_score'] = (summary['utmos_mean'] * 0.7 + normalized_pesq * 0.3)
    
    # Calculate an overall intelligibility score (using STOI and SDR metrics)
    if 'stoi_mean' in summary and 'sdr_mean' in summary:
        # Normalize SDR to 0-1 range (typical range might be -30 to 0 dB)
        # Higher SDR is better, but typical values are negative
        normalized_sdr = min(1.0, max(0.0, (summary['sdr_mean'] + 35) / 35))
        # STOI is already in 0-1 range
        summary['intelligibility_score'] = (summary['stoi_mean'] * 0.6 + normalized_sdr * 0.4)
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Process TTS evaluation results and calculate metrics")
    parser.add_argument("--independent", type=Path, required=True, 
                        help="Path to independent metrics file (line-by-line JSON format)")
    parser.add_argument("--dependent", type=Path, required=True, 
                        help="Path to dependent metrics file (line-by-line JSON format)")
    parser.add_argument("--output", type=Path, required=True, 
                        help="Path to output directory")
    parser.add_argument("--verbose", "-v", action="count", default=0,
                       help="Increase verbosity (-v: INFO, -vv: DEBUG)")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.WARNING
    if args.verbose == 1:
        log_level = logging.INFO
    elif args.verbose >= 2:
        log_level = logging.DEBUG
    
    logging.basicConfig(level=log_level, 
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Read data
    logging.info(f"Reading independent metrics from {args.independent}")
    independent_data = read_line_by_line_json(args.independent)
    logging.info(f"Found {len(independent_data)} entries in independent metrics")
    
    logging.info(f"Reading dependent metrics from {args.dependent}")
    dependent_data = read_line_by_line_json(args.dependent)
    logging.info(f"Found {len(dependent_data)} entries in dependent metrics")
    
    # Write standardized JSONL
    write_jsonl(independent_data, args.output / "independent.jsonl")
    write_jsonl(dependent_data, args.output / "dependent.jsonl")
    
    # Calculate summary statistics
    summary = calculate_metrics_summary(independent_data, dependent_data)
    
    # Save summary
    with open(args.output / "metrics_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    logging.info(f"Summary metrics saved to {args.output / 'metrics_summary.json'}")
    
    # Print summary to console
    print("\n===== TTS Evaluation Summary =====")
    for metric, value in summary.items():
        print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()


# python src/summarizer.py --independent evaluation_results/independent.json --dependent evaluation_results/dependent.json --output evaluation_results    