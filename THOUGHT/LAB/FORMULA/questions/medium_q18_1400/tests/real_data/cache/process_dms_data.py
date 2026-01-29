#!/usr/bin/env python3
"""
Process MaveDB DMS data for Q18 analysis.

This script:
1. Reads raw CSV scores from MaveDB
2. Filters to single amino acid substitutions only
3. Parses HGVS protein notation (e.g., p.Ala279Tyr)
4. Saves in standardized JSON format

Data source: MaveDB BRCA1 RING domain DMS data
URN: urn:mavedb:00000003-a-2
"""

import csv
import json
import re
from pathlib import Path

# Amino acid code mapping
AA_CODES = {
    'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',
    'Gln': 'Q', 'Glu': 'E', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
    'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
    'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V',
    'Ter': '*'  # Stop codon
}


def parse_hgvs_protein(hgvs_pro):
    """
    Parse HGVS protein notation.

    Examples:
        p.Ala279Tyr -> {'position': 279, 'wt': 'A', 'mut': 'Y'}
        p.[Gly210Thr;Ala224Thr] -> None (multi-mutation)

    Returns None for invalid/multi-mutation entries.
    """
    if not hgvs_pro or hgvs_pro == 'NA':
        return None

    # Skip multi-mutations (contain brackets or semicolons)
    if '[' in hgvs_pro or ';' in hgvs_pro:
        return None

    # Pattern for single substitution: p.AaaNNNBbb
    # where Aaa = 3-letter wt, NNN = position, Bbb = 3-letter mutant
    pattern = r'^p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})$'
    match = re.match(pattern, hgvs_pro)

    if not match:
        return None

    wt_3letter, position, mut_3letter = match.groups()

    # Convert to single-letter codes
    wt = AA_CODES.get(wt_3letter)
    mut = AA_CODES.get(mut_3letter)

    if not wt or not mut:
        return None

    return {
        'position': int(position),
        'wt': wt,
        'mut': mut
    }


def process_brca1_dms():
    """Process BRCA1 RING domain DMS data."""

    cache_dir = Path(__file__).parent
    input_file = cache_dir / 'brca1_raw_scores.csv'
    output_file = cache_dir / 'dms_data.json'

    if not input_file.exists():
        print(f"ERROR: Input file not found: {input_file}")
        print("Run the curl command first to download the raw data.")
        return None

    # Read raw CSV
    mutations = []
    skipped_multi = 0
    skipped_no_score = 0
    skipped_parse_error = 0

    with open(input_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            hgvs_pro = row.get('hgvs_pro', '')
            score_str = row.get('score', '')

            # Parse HGVS notation
            parsed = parse_hgvs_protein(hgvs_pro)

            if not parsed:
                if '[' in hgvs_pro or ';' in hgvs_pro:
                    skipped_multi += 1
                else:
                    skipped_parse_error += 1
                continue

            # Check for valid score
            if not score_str or score_str == 'None' or score_str == 'NA':
                skipped_no_score += 1
                continue

            try:
                score = float(score_str)
            except ValueError:
                skipped_no_score += 1
                continue

            # Also get standard error if available
            se_str = row.get('SE', '')
            se = None
            if se_str and se_str != 'None' and se_str != 'NA':
                try:
                    se = float(se_str)
                except ValueError:
                    pass

            mutations.append({
                'position': parsed['position'],
                'wt': parsed['wt'],
                'mut': parsed['mut'],
                'fitness': score,
                'se': se
            })

    print(f"\nProcessing summary:")
    print(f"  Single AA mutations with scores: {len(mutations)}")
    print(f"  Skipped (multi-mutation): {skipped_multi}")
    print(f"  Skipped (no/invalid score): {skipped_no_score}")
    print(f"  Skipped (parse error): {skipped_parse_error}")

    # Sort by position
    mutations.sort(key=lambda m: (m['position'], m['wt'], m['mut']))

    # BRCA1 RING domain sequence from MaveDB
    # This is the target sequence from the dataset metadata
    dna_sequence = (
        "GATTTATCTGCTCTTCGCGTTGAAGAAGTACAAAATGTCATTAATGCTATGCAGAAAATCTTAGAGTGTCC"
        "CATCTGCCTGGAGTTGATCAAGGAACCTGTCTCCACAAAGTGTGACCACATATTTTGCAAATTTTGCATGC"
        "TGAAACTTCTCAACCAGAAGAAAGGGCCTTCACAGTGTCCTTTATGTAAGAATGATATAACCAAAAGGAGC"
        "CTACAAGAAAGTACGAGATTTAGTCAACTTGTTGAAGAGCTATTGAAAATCATTTGTGCTTTTCAGCTTGA"
        "CACAGGTTTGGAGTATGCAAACAGCTATAATTTTGCAAAAAAGGAAAATAACTCTCCTGAACATCTAAAAG"
        "ATGAAGTTTCTATCATCCAAAGTATGGGCTACAGAAACCGTGCCAAAAGACTTCTACAGAGTGAACCCGAA"
        "AATCCTTCCTTGCAGGAAACCAGTCTCAGTGTCCAACTCTCTAACCTTGGAACTGTGAGAACTCTGAGGAC"
        "AAAGCAGCGGATACAACCTCAAAGGACGTCTGTCTACATTGAATTGGGATCTGATTCTTCTGAAGATACCG"
        "TTAATAAGGCAACTTATTGCAGTGTGGGAGATCAAGAATTGTTACAAATCACCCCTCAAGGAACCAGGGAT"
        "GAAATCAGTTTGGATTCTGCAAAAAAGGCTGCTTGTGAATTTTCTGAGACGGATGTAACAAATACTGAACA"
        "TCATCAACCCAGTAATAATGATTTGAACACCACTGAGAAGCGTGCAGCTGAGAGGCATCCAGAAAAGTATC"
        "AGGGTAGTTCTGTTTCAAACTTGCATGTGGAGCCATGTGGCACAAATACTCATGCCAGCTCATTACAGCAT"
        "GAGAACAGCAGTTTATTACTCACTAAAGACAGAATGAATGTAGAAAAGGCTGAGTTC"
    )

    # Translate to protein
    codon_table = {
        'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
        'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
        'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
        'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
        'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
        'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
        'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
        'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
        'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
        'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
        'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
        'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
        'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
        'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
        'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
        'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
    }

    protein_seq = ''
    for i in range(0, len(dna_sequence) - 2, 3):
        codon = dna_sequence[i:i+3]
        aa = codon_table.get(codon, 'X')
        if aa == '*':
            break
        protein_seq += aa

    # Create output data structure
    output_data = {
        'protein': 'BRCA1 RING domain',
        'uniprot_id': 'P38398',
        'sequence': protein_seq,
        'sequence_length': len(protein_seq),
        'source': 'MaveDB',
        'mavedb_urn': 'urn:mavedb:00000003-a-2',
        'assay': 'E3 ubiquitin ligase activity (phage autoubiquitination)',
        'publication': 'Starita et al. 2015 (PMID: 25823446)',
        'score_interpretation': 'Higher scores = more functional (WT-like), lower = loss of function',
        'n_mutations': len(mutations),
        'mutations': mutations
    }

    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved to: {output_file}")
    print(f"  Protein: {output_data['protein']}")
    print(f"  Sequence length: {len(protein_seq)} residues")
    print(f"  Total mutations: {len(mutations)}")

    # Print score distribution
    scores = [m['fitness'] for m in mutations]
    print(f"\nScore distribution:")
    print(f"  Min: {min(scores):.3f}")
    print(f"  Max: {max(scores):.3f}")
    print(f"  Mean: {sum(scores)/len(scores):.3f}")

    return output_data


def process_ube2i_dms():
    """Process UBE2I DMS data."""

    cache_dir = Path(__file__).parent
    input_file = cache_dir / 'ube2i_raw_scores.csv'
    output_file = cache_dir / 'dms_data_ube2i.json'

    if not input_file.exists():
        print(f"ERROR: Input file not found: {input_file}")
        return None

    # Read raw CSV
    mutations = []
    skipped_multi = 0
    skipped_no_score = 0
    skipped_parse_error = 0

    with open(input_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            hgvs_pro = row.get('hgvs_pro', '')
            score_str = row.get('score', '')

            # Parse HGVS notation
            parsed = parse_hgvs_protein(hgvs_pro)

            if not parsed:
                if '[' in hgvs_pro or ';' in hgvs_pro:
                    skipped_multi += 1
                else:
                    skipped_parse_error += 1
                continue

            # Check for valid score
            if not score_str or score_str == 'None' or score_str == 'NA':
                skipped_no_score += 1
                continue

            try:
                score = float(score_str)
            except ValueError:
                skipped_no_score += 1
                continue

            # Also get standard error if available
            se_str = row.get('se', '') or row.get('SE', '')
            se = None
            if se_str and se_str != 'None' and se_str != 'NA':
                try:
                    se = float(se_str)
                except ValueError:
                    pass

            mutations.append({
                'position': parsed['position'],
                'wt': parsed['wt'],
                'mut': parsed['mut'],
                'fitness': score,
                'se': se
            })

    print(f"\nProcessing UBE2I summary:")
    print(f"  Single AA mutations with scores: {len(mutations)}")
    print(f"  Skipped (multi-mutation): {skipped_multi}")
    print(f"  Skipped (no/invalid score): {skipped_no_score}")
    print(f"  Skipped (parse error): {skipped_parse_error}")

    # Sort by position
    mutations.sort(key=lambda m: (m['position'], m['wt'], m['mut']))

    # UBE2I sequence from UniProt P63279
    protein_seq = (
        "MSGIALSRLAQERKAWRKDHPFGFVAVPTKNPDGTMNLMNWECAIPGKKGTP"
        "WEGGLFKLRMLFKDDYPSSPPKCKFEPPLFHPNVYPSGTVCLSILEEDKDWRPAIT"
        "IKQILLGIQELLNEPNIQDPAQAEAYTIYCQNRVEYEKRVRAQAKKFAPS"
    )

    # Create output data structure
    output_data = {
        'protein': 'UBE2I (SUMO E2 conjugase)',
        'uniprot_id': 'P63279',
        'sequence': protein_seq,
        'sequence_length': len(protein_seq),
        'source': 'MaveDB',
        'mavedb_urn': 'urn:mavedb:00000001-a-1',
        'assay': 'Yeast complementation (functional fitness)',
        'publication': 'Weile et al. 2017 (PMID: 29269382)',
        'score_interpretation': 'Higher scores = more functional (WT-like), lower = loss of function. Normalized 0-1 scale.',
        'n_mutations': len(mutations),
        'mutations': mutations
    }

    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved to: {output_file}")
    print(f"  Protein: {output_data['protein']}")
    print(f"  Sequence length: {len(protein_seq)} residues")
    print(f"  Total mutations: {len(mutations)}")

    # Print score distribution
    scores = [m['fitness'] for m in mutations]
    print(f"\nScore distribution:")
    print(f"  Min: {min(scores):.3f}")
    print(f"  Max: {max(scores):.3f}")
    print(f"  Mean: {sum(scores)/len(scores):.3f}")

    return output_data


def process_tp53_dms():
    """Process TP53 DMS data."""

    cache_dir = Path(__file__).parent
    input_file = cache_dir / 'tp53_raw_scores.csv'
    output_file = cache_dir / 'dms_data_tp53.json'

    if not input_file.exists():
        print(f"ERROR: Input file not found: {input_file}")
        return None

    # Read raw CSV
    mutations = []
    skipped_multi = 0
    skipped_no_score = 0
    skipped_parse_error = 0

    with open(input_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            hgvs_pro = row.get('hgvs_pro', '')
            score_str = row.get('score', '')

            # Parse HGVS notation
            parsed = parse_hgvs_protein(hgvs_pro)

            if not parsed:
                if '[' in hgvs_pro or ';' in hgvs_pro:
                    skipped_multi += 1
                else:
                    skipped_parse_error += 1
                continue

            # Check for valid score
            if not score_str or score_str == 'None' or score_str == 'NA':
                skipped_no_score += 1
                continue

            try:
                score = float(score_str)
            except ValueError:
                skipped_no_score += 1
                continue

            mutations.append({
                'position': parsed['position'],
                'wt': parsed['wt'],
                'mut': parsed['mut'],
                'fitness': score,
                'se': None  # TP53 data doesn't have SE
            })

    print(f"\nProcessing TP53 summary:")
    print(f"  Single AA mutations with scores: {len(mutations)}")
    print(f"  Skipped (multi-mutation): {skipped_multi}")
    print(f"  Skipped (no/invalid score): {skipped_no_score}")
    print(f"  Skipped (parse error): {skipped_parse_error}")

    # Sort by position
    mutations.sort(key=lambda m: (m['position'], m['wt'], m['mut']))

    # TP53 sequence from UniProt P04637
    protein_seq = (
        "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPAL"
        "NKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDS"
        "SGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD"
    )

    # Create output data structure
    output_data = {
        'protein': 'TP53 (p53 tumor suppressor)',
        'uniprot_id': 'P04637',
        'sequence': protein_seq,
        'sequence_length': len(protein_seq),
        'source': 'MaveDB',
        'mavedb_urn': 'urn:mavedb:00001234-e-1',
        'assay': 'Yeast-based transcriptional activity (p53AIP1 promoter)',
        'publication': 'Kato et al. 2003 (PMID: 12826609)',
        'score_interpretation': 'Higher scores = more functional (WT-like = 100%), lower = loss of function (0% = null)',
        'n_mutations': len(mutations),
        'mutations': mutations
    }

    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved to: {output_file}")
    print(f"  Protein: {output_data['protein']}")
    print(f"  Sequence length: {len(protein_seq)} residues")
    print(f"  Total mutations: {len(mutations)}")

    # Print score distribution
    scores = [m['fitness'] for m in mutations]
    print(f"\nScore distribution:")
    print(f"  Min: {min(scores):.3f}")
    print(f"  Max: {max(scores):.3f}")
    print(f"  Mean: {sum(scores)/len(scores):.3f}")

    return output_data


if __name__ == '__main__':
    print("=" * 60)
    print("Processing BRCA1 RING domain DMS data...")
    print("=" * 60)
    process_brca1_dms()

    print("\n" + "=" * 60)
    print("Processing UBE2I DMS data...")
    print("=" * 60)
    process_ube2i_dms()

    print("\n" + "=" * 60)
    print("Processing TP53 DMS data...")
    print("=" * 60)
    process_tp53_dms()
