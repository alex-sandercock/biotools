#!/usr/bin/env python3
"""
Simulate Realistic GWAS Diversity Panel: Atlantic Giant Pumpkin
================================================================
Version 3: Realistic QTL peak with LD structure

Changes from v2:
- ONE causal QTL SNP that affects phenotype
- 2 nearby tag SNPs within 1.5 Mb in HIGH LD (r² > 0.8) with causal SNP
- Tag SNPs appear significant due to LD, not because they're causal
- After LD filtering, one candidate region remains
"""

import numpy as np
import pandas as pd
from datetime import datetime
import random

# Set seed for reproducibility
np.random.seed(2024)
random.seed(2024)

# =============================================================================
# Parameters
# =============================================================================
N_SAMPLES = 200

N_GOOD_SNPS = 1000
N_LOW_MAF_SNPS = 100
N_HIGH_MISSING_SNPS = 100
N_TOTAL_SNPS = N_GOOD_SNPS + N_LOW_MAF_SNPS + N_HIGH_MISSING_SNPS

N_CHROMOSOMES = 20

# QTL parameters - NOW JUST ONE CAUSAL SNP + TAG SNPS IN LD
QTL_CHROMOSOME = 4
QTL_EFFECT_SIZE = 150  # pounds per allele (increased since only 1 causal SNP now)
N_TAG_SNPS = 2  # SNPs in LD with causal SNP (within 1.5 Mb)
LD_R2_TARGET = 0.85  # Target LD between causal and tag SNPs

# Population structure parameters
N_ANCESTRAL_POPS = 3

# =============================================================================
# Define Geographic Regions and States
# =============================================================================

REGIONS = {
    "Eastern": {
        "states": ["New York", "Pennsylvania", "Ohio", "Rhode Island", 
                   "Connecticut", "Massachusetts", "New Jersey"],
        "base_ancestry": [0.8, 0.15, 0.05],
        "base_weight": 950,
    },
    "Midwest": {
        "states": ["Wisconsin", "Iowa", "Illinois", "Indiana", "Michigan", 
                   "Minnesota", "Nebraska", "Missouri"],
        "base_ancestry": [0.2, 0.7, 0.1],
        "base_weight": 850,
    },
    "Western": {
        "states": ["California", "Oregon", "Washington", "Colorado", "Arizona"],
        "base_ancestry": [0.1, 0.2, 0.7],
        "base_weight": 750,
    }
}

STATE_TO_REGION = {}
for region, info in REGIONS.items():
    for state in info["states"]:
        STATE_TO_REGION[state] = region

# =============================================================================
# Helper Functions
# =============================================================================

def generate_ancestry_proportions(region, n_samples):
    base = np.array(REGIONS[region]["base_ancestry"])
    ancestries = []
    for _ in range(n_samples):
        concentration = 8
        noisy = np.random.dirichlet(base * concentration)
        ancestries.append(noisy)
    return np.array(ancestries)


def simulate_genotype_with_structure(ancestral_freqs, ancestry_props):
    p = np.dot(ancestry_props, ancestral_freqs)
    p = np.clip(p, 0.001, 0.999)
    genotype = np.random.binomial(2, p)
    return genotype


def generate_ld_genotypes(causal_genotypes, target_r2=0.85):
    """
    Generate tag SNP genotypes in LD with causal SNP.
    Uses a probabilistic approach to maintain target r².
    """
    n = len(causal_genotypes)
    tag_genotypes = np.zeros(n)
    
    # Probability of copying vs randomizing
    # Higher prob_copy = higher LD
    prob_copy = np.sqrt(target_r2)  # Approximate relationship
    
    for i in range(n):
        if np.isnan(causal_genotypes[i]):
            tag_genotypes[i] = np.nan
        elif np.random.random() < prob_copy:
            # Copy genotype (maintains LD)
            tag_genotypes[i] = causal_genotypes[i]
        else:
            # Introduce recombination/noise
            # Randomly shift genotype by -1, 0, or +1
            shift = np.random.choice([-1, 0, 1], p=[0.2, 0.6, 0.2])
            tag_genotypes[i] = np.clip(causal_genotypes[i] + shift, 0, 2)
    
    return tag_genotypes


def calculate_r2(geno1, geno2):
    """Calculate r² between two genotype vectors."""
    # Remove missing data
    mask = ~(np.isnan(geno1) | np.isnan(geno2))
    g1 = geno1[mask]
    g2 = geno2[mask]
    
    if len(g1) < 10:
        return np.nan
    
    # Correlation coefficient squared
    r = np.corrcoef(g1, g2)[0, 1]
    return r ** 2


def generate_depth(n, mean_depth=30, sd_depth=10):
    depths = np.maximum(5, np.round(np.random.normal(mean_depth, sd_depth, n))).astype(int)
    return depths


def generate_ad(genotype, total_depth):
    if np.isnan(genotype) or np.isnan(total_depth):
        return None, None
    expected_ratio = genotype / 2
    actual_ratio = np.clip(np.random.normal(expected_ratio, 0.05), 0, 1)
    alt_depth = int(round(total_depth * actual_ratio))
    ref_depth = int(total_depth - alt_depth)
    return ref_depth, alt_depth


def gt_to_vcf(gt):
    if np.isnan(gt):
        return "./."
    gt = int(gt)
    if gt == 0:
        return "0/0"
    elif gt == 1:
        return "0/1"
    elif gt == 2:
        return "1/1"
    return "./."


# =============================================================================
# Generate Sample Metadata
# =============================================================================
print("Generating sample metadata with geographic structure...")

region_assignments = (["Eastern"] * 70 + ["Midwest"] * 70 + ["Western"] * 60)
random.shuffle(region_assignments)

states = []
for region in region_assignments:
    state = random.choice(REGIONS[region]["states"])
    states.append(state)

ancestry_matrix = np.zeros((N_SAMPLES, N_ANCESTRAL_POPS))
for i, region in enumerate(region_assignments):
    ancestry_matrix[i, :] = generate_ancestry_proportions(region, 1)[0]

sample_ids = [f"AGP_{i:03d}" for i in range(1, N_SAMPLES + 1)]

breeding_programs = []
for state in states:
    if state in ["New York", "Pennsylvania", "Ohio"]:
        programs = ["Cornell_Pumpkin", "PSU_Cucurbit", "OSU_Giant", "Private_Breeder"]
    elif state in ["Wisconsin", "Iowa", "Illinois"]:
        programs = ["UW_Cucurbit", "ISU_Horticulture", "UI_Pumpkin", "Midwest_Giant_Assoc"]
    elif state in ["California", "Oregon", "Washington"]:
        programs = ["UCD_Cucurbit", "OSU_Veg", "WSU_Horticulture", "West_Coast_Seeds"]
    else:
        programs = ["University_Program", "Private_Breeder", "Grower_Selection"]
    breeding_programs.append(random.choice(programs))

sequencing_years = np.random.choice([2021, 2022, 2023, 2024], N_SAMPLES, p=[0.1, 0.2, 0.3, 0.4])
lane_ids = [f"Lane_{np.random.randint(1, 9)}" for _ in range(N_SAMPLES)]

plates = []
wells = []
for i in range(N_SAMPLES):
    plate_num = i // 96 + 1
    well_idx = i % 96
    row = "ABCDEFGH"[well_idx % 8]
    col = well_idx // 8 + 1
    plates.append(f"Plate_{plate_num}")
    wells.append(f"{row}{col}")

# =============================================================================
# Generate SNP Information
# =============================================================================
print("Generating SNP information...")

snps_per_chrom = int(np.ceil(N_TOTAL_SNPS / N_CHROMOSOMES))

chromosomes = []
positions = []
snp_ids = []
refs = []
alts = []
quals = []

nucleotides = ["A", "C", "G", "T"]

snp_counter = 0
for chrom_num in range(1, N_CHROMOSOMES + 1):
    chrom_name = f"Chr{chrom_num:02d}"
    n_snps_this_chrom = min(snps_per_chrom, N_TOTAL_SNPS - snp_counter)
    
    if n_snps_this_chrom <= 0:
        break
    
    # Realistic C. maxima chromosome lengths based on HZAU genome assembly 
    # (Zeng et al. 2024, Plant Communications; 345.14 Mb total, N50=19.03 Mb)
    # Estimated from total genome size and N50, ranging from ~13 to ~22 Mb
    chrom_lengths_mb = [21.8, 20.5, 19.8, 19.2, 18.7, 18.3, 17.9, 17.5, 17.1, 16.8,
                        16.5, 16.2, 15.9, 15.6, 15.3, 15.0, 14.7, 14.4, 14.1, 13.8]
    chrom_length = int(chrom_lengths_mb[chrom_num - 1] * 1_000_000)
    chrom_positions = sorted(np.random.choice(range(1, chrom_length), n_snps_this_chrom, replace=False))
    
    for pos in chrom_positions:
        snp_counter += 1
        chromosomes.append(chrom_name)
        positions.append(pos)
        snp_ids.append(f"SNP_{snp_counter:04d}")
        
        ref = random.choice(nucleotides)
        refs.append(ref)
        alts.append(random.choice([n for n in nucleotides if n != ref]))
        quals.append(np.random.randint(100, 999))

# Randomly assign quality categories across all SNPs
all_indices = list(range(N_TOTAL_SNPS))
random.shuffle(all_indices)

low_maf_indices = set(all_indices[:N_LOW_MAF_SNPS])
high_missing_indices = set(all_indices[N_LOW_MAF_SNPS:N_LOW_MAF_SNPS + N_HIGH_MISSING_SNPS])

quality_categories = []
for i in range(N_TOTAL_SNPS):
    if i in low_maf_indices:
        quality_categories.append("low_maf")
    elif i in high_missing_indices:
        quality_categories.append("high_missing")
    else:
        quality_categories.append("good")

snp_info = pd.DataFrame({
    "CHROM": chromosomes,
    "POS": positions,
    "ID": snp_ids,
    "REF": refs,
    "ALT": alts,
    "QUAL": quals,
    "quality_category": quality_categories
})

# =============================================================================
# Identify QTL region: 1 causal SNP + 2 tag SNPs within 1.5 Mb
# Target region: ~19.5 Mb on Chr04 (based on real Atlantic Giant fruit size QTL)
# =============================================================================
print("\nSetting up QTL region with LD structure...")

qtl_chrom = f"Chr{QTL_CHROMOSOME:02d}"
TARGET_POSITION = 19_500_000  # Target ~19.5 Mb region

# Find good quality SNPs on QTL chromosome
good_snps_on_qtl_chrom = snp_info[
    (snp_info["CHROM"] == qtl_chrom) & 
    (snp_info["quality_category"] == "good")
].copy()

good_snps_on_qtl_chrom = good_snps_on_qtl_chrom.sort_values("POS").reset_index(drop=False)
good_snps_on_qtl_chrom.rename(columns={"index": "original_index"}, inplace=True)

# Find a causal SNP closest to target position that has at least 2 other good SNPs within 1.5 Mb
# Sort candidates by distance to target position
good_snps_on_qtl_chrom["dist_to_target"] = abs(good_snps_on_qtl_chrom["POS"] - TARGET_POSITION)
good_snps_on_qtl_chrom = good_snps_on_qtl_chrom.sort_values("dist_to_target")

causal_snp_idx = None
tag_snp_indices = []

for _, row in good_snps_on_qtl_chrom.iterrows():
    causal_pos = row["POS"]
    
    # Find SNPs within 1.5 Mb
    nearby = good_snps_on_qtl_chrom[
        (abs(good_snps_on_qtl_chrom["POS"] - causal_pos) <= 1500000) &
        (good_snps_on_qtl_chrom["POS"] != causal_pos)
    ]
    
    if len(nearby) >= N_TAG_SNPS:
        causal_snp_idx = row["original_index"]
        # Pick closest SNPs as tag SNPs
        nearby_sorted = nearby.iloc[(nearby["POS"] - causal_pos).abs().argsort()]
        tag_snp_indices = nearby_sorted.head(N_TAG_SNPS)["original_index"].tolist()
        break

if causal_snp_idx is None:
    raise ValueError("Could not find suitable QTL region!")

# Store QTL information
qtl_causal_index = causal_snp_idx
qtl_tag_indices = tag_snp_indices
all_qtl_indices = [qtl_causal_index] + qtl_tag_indices

print(f"  Causal SNP: {snp_info.loc[qtl_causal_index, 'ID']} at position {snp_info.loc[qtl_causal_index, 'POS']:,}")
for tag_idx in qtl_tag_indices:
    distance = abs(snp_info.loc[tag_idx, "POS"] - snp_info.loc[qtl_causal_index, "POS"])
    print(f"  Tag SNP: {snp_info.loc[tag_idx, 'ID']} at position {snp_info.loc[tag_idx, 'POS']:,} ({distance/1000:.1f} kb away)")

# Update SNP info with QTL annotations
snp_info["is_QTL"] = False
snp_info["QTL_role"] = "none"

snp_info.loc[qtl_causal_index, "is_QTL"] = True
snp_info.loc[qtl_causal_index, "QTL_role"] = "causal"

for tag_idx in qtl_tag_indices:
    snp_info.loc[tag_idx, "is_QTL"] = True
    snp_info.loc[tag_idx, "QTL_role"] = "tag_LD"

# =============================================================================
# Generate Ancestral Allele Frequencies
# =============================================================================
print("\nGenerating ancestral population allele frequencies...")

ancestral_freqs = np.zeros((N_TOTAL_SNPS, N_ANCESTRAL_POPS))

for i in range(N_TOTAL_SNPS):
    quality = snp_info.loc[i, "quality_category"]
    
    if quality == "low_maf":
        base_freq = np.random.uniform(0.001, 0.02)
        ancestral_freqs[i, :] = base_freq + np.random.uniform(-0.005, 0.005, N_ANCESTRAL_POPS)
        ancestral_freqs[i, :] = np.clip(ancestral_freqs[i, :], 0.001, 0.05)
        
    elif i == qtl_causal_index:
        # Causal QTL: similar frequency across populations (not confounded)
        base_freq = np.random.uniform(0.35, 0.50)
        ancestral_freqs[i, :] = base_freq + np.random.uniform(-0.03, 0.03, N_ANCESTRAL_POPS)
        ancestral_freqs[i, :] = np.clip(ancestral_freqs[i, :], 0.15, 0.85)
        
    elif i in qtl_tag_indices:
        # Tag SNPs will get genotypes from LD, but need placeholder frequencies
        # These won't be used directly since we'll copy from causal
        ancestral_freqs[i, :] = ancestral_freqs[qtl_causal_index, :]
        
    else:
        base_freq = np.random.uniform(0.1, 0.9)
        fst = np.random.beta(2, 5)
        diff_scale = fst * 0.4
        diffs = np.random.uniform(-diff_scale, diff_scale, N_ANCESTRAL_POPS)
        diffs = diffs - np.mean(diffs)
        ancestral_freqs[i, :] = base_freq + diffs
        ancestral_freqs[i, :] = np.clip(ancestral_freqs[i, :], 0.05, 0.95)

# =============================================================================
# Generate Genotype Matrix
# =============================================================================
print("Generating genotype matrix...")

geno_matrix = np.full((N_TOTAL_SNPS, N_SAMPLES), np.nan)

# First pass: generate all non-tag SNP genotypes
for i in range(N_TOTAL_SNPS):
    if i in qtl_tag_indices:
        continue  # Will generate these after causal SNP
    
    quality = snp_info.loc[i, "quality_category"]
    
    for j in range(N_SAMPLES):
        if quality == "high_missing" and np.random.random() < np.random.uniform(0.5, 0.8):
            continue
        
        geno_matrix[i, j] = simulate_genotype_with_structure(
            ancestral_freqs[i, :], 
            ancestry_matrix[j, :]
        )
    
    if quality == "good":
        missing_rate = np.random.uniform(0.01, 0.05)
        missing_idx = np.random.choice(N_SAMPLES, int(N_SAMPLES * missing_rate), replace=False)
        geno_matrix[i, missing_idx] = np.nan
    
    if (i + 1) % 300 == 0:
        print(f"  Generated {i + 1}/{N_TOTAL_SNPS} SNPs...")

# Second pass: generate tag SNP genotypes in LD with causal SNP
print("\nGenerating tag SNPs in LD with causal SNP...")
causal_genotypes = geno_matrix[qtl_causal_index, :]

for tag_idx in qtl_tag_indices:
    # Generate genotypes in high LD with causal SNP
    tag_genotypes = generate_ld_genotypes(causal_genotypes, target_r2=LD_R2_TARGET)
    geno_matrix[tag_idx, :] = tag_genotypes
    
    # Calculate actual r²
    actual_r2 = calculate_r2(causal_genotypes, tag_genotypes)
    print(f"  {snp_info.loc[tag_idx, 'ID']}: r² = {actual_r2:.3f} with causal SNP")
    
    # Update ancestral frequencies based on regenerated genotypes
    # Calculate frequency per population based on actual genotypes
    for pop_idx, pop in enumerate(["Eastern", "Midwest", "Western"]):
        pop_mask = np.array(region_assignments) == pop
        pop_genos = tag_genotypes[pop_mask]
        valid_genos = pop_genos[~np.isnan(pop_genos)]
        if len(valid_genos) > 0:
            ancestral_freqs[tag_idx, pop_idx] = np.mean(valid_genos) / 2

# =============================================================================
# Generate Depth and Allelic Depth Data
# =============================================================================
print("\nGenerating depth information...")

depth_matrix = np.full((N_TOTAL_SNPS, N_SAMPLES), np.nan)
ad_ref_matrix = np.full((N_TOTAL_SNPS, N_SAMPLES), np.nan)
ad_alt_matrix = np.full((N_TOTAL_SNPS, N_SAMPLES), np.nan)

for i in range(N_TOTAL_SNPS):
    quality = snp_info.loc[i, "quality_category"]
    
    for j in range(N_SAMPLES):
        if not np.isnan(geno_matrix[i, j]):
            if quality == "high_missing":
                depth_matrix[i, j] = generate_depth(1, mean_depth=12, sd_depth=6)[0]
            else:
                depth_matrix[i, j] = generate_depth(1, mean_depth=30, sd_depth=10)[0]
            
            ad_ref, ad_alt = generate_ad(geno_matrix[i, j], depth_matrix[i, j])
            if ad_ref is not None:
                ad_ref_matrix[i, j] = ad_ref
                ad_alt_matrix[i, j] = ad_alt

    if (i + 1) % 300 == 0:
        print(f"  Generated depth for {i + 1}/{N_TOTAL_SNPS} SNPs...")

# =============================================================================
# Generate Phenotype Data - ONLY CAUSAL SNP AFFECTS PHENOTYPE
# =============================================================================
print("\nGenerating phenotype data (only causal SNP affects trait)...")

fruit_weight = np.zeros(N_SAMPLES)

for j in range(N_SAMPLES):
    # Population structure effect
    pop_effect = (
        ancestry_matrix[j, 0] * REGIONS["Eastern"]["base_weight"] +
        ancestry_matrix[j, 1] * REGIONS["Midwest"]["base_weight"] +
        ancestry_matrix[j, 2] * REGIONS["Western"]["base_weight"]
    )
    
    # QTL effect - ONLY from causal SNP
    causal_gt = geno_matrix[qtl_causal_index, j]
    if np.isnan(causal_gt):
        causal_gt = 1  # Impute as het
    qtl_effect = causal_gt * QTL_EFFECT_SIZE
    
    # Polygenic background
    good_snp_indices = [idx for idx in range(N_TOTAL_SNPS) 
                        if snp_info.loc[idx, "quality_category"] == "good" 
                        and idx not in all_qtl_indices]
    polygenic_snps = random.sample(good_snp_indices[:500], min(50, len(good_snp_indices)))
    polygenic_effect = 0
    for snp_idx in polygenic_snps:
        gt = geno_matrix[snp_idx, j]
        if not np.isnan(gt):
            polygenic_effect += (gt - 1) * np.random.uniform(2, 5)
    
    # Environmental noise
    env_noise = np.random.normal(0, 60)
    
    weight = pop_effect + qtl_effect + polygenic_effect + env_noise
    fruit_weight[j] = max(200, round(weight, 1))

# =============================================================================
# Create Phenotype/Passport Data Frame
# =============================================================================
passport_data = pd.DataFrame({
    "Sample_ID": sample_ids,
    "State": states,
    "Region": region_assignments,
    "Breeding_Program": breeding_programs,
    "Sequencing_Year": sequencing_years,
    "Sequencing_Lane": lane_ids,
    "Plate": plates,
    "Well": wells,
    "Eastern_Ancestry": np.round(ancestry_matrix[:, 0], 3),
    "Midwest_Ancestry": np.round(ancestry_matrix[:, 1], 3),
    "Western_Ancestry": np.round(ancestry_matrix[:, 2], 3),
    "Fruit_Weight_lbs": fruit_weight
})

# =============================================================================
# Write VCF File
# =============================================================================
print("\nWriting VCF file...")

vcf_file = "/home/claude/atlantic_giant_pumpkin_diversity.vcf"

with open(vcf_file, "w") as f:
    f.write("##fileformat=VCFv4.2\n")
    f.write(f"##fileDate={datetime.now().strftime('%Y%m%d')}\n")
    f.write("##source=BIGapp_Workshop_Diversity_Panel_v3_LD_QTL\n")
    f.write("##reference=Cucurbita_maxima_HZAU_T2T_v1\n")
    f.write("##assembly=Zeng_et_al_2024_Plant_Commun_doi:10.1016/j.xplc.2024.100935\n")
    
    # Chromosome lengths for C. maxima HZAU genome (345.14 Mb total, T2T assembly)
    chrom_lengths = [21800000, 20500000, 19800000, 19200000, 18700000, 18300000, 17900000,
                     17500000, 17100000, 16800000, 16500000, 16200000, 15900000, 15600000,
                     15300000, 15000000, 14700000, 14400000, 14100000, 13800000]
    for chrom_num in range(1, N_CHROMOSOMES + 1):
        f.write(f"##contig=<ID=Chr{chrom_num:02d},length={chrom_lengths[chrom_num-1]}>\n")
    
    f.write('##INFO=<ID=NS,Number=1,Type=Integer,Description="Number of Samples With Data">\n')
    f.write('##INFO=<ID=DP,Number=1,Type=Integer,Description="Total Depth">\n')
    f.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
    f.write('##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read Depth">\n')
    f.write('##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allelic depths for the ref and alt alleles">\n')
    
    f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t")
    f.write("\t".join(sample_ids) + "\n")
    
    for i in range(N_TOTAL_SNPS):
        ns = int(np.sum(~np.isnan(geno_matrix[i, :])))
        total_dp = int(np.nansum(depth_matrix[i, :]))
        
        info_field = f"NS={ns};DP={total_dp}"
        
        sample_fields = []
        for j in range(N_SAMPLES):
            gt = gt_to_vcf(geno_matrix[i, j])
            
            if np.isnan(geno_matrix[i, j]):
                sample_fields.append("./.:.:.,.")
            else:
                dp = int(depth_matrix[i, j])
                ad_ref = int(ad_ref_matrix[i, j])
                ad_alt = int(ad_alt_matrix[i, j])
                sample_fields.append(f"{gt}:{dp}:{ad_ref},{ad_alt}")
        
        line = "\t".join([
            snp_info.loc[i, "CHROM"],
            str(snp_info.loc[i, "POS"]),
            snp_info.loc[i, "ID"],
            snp_info.loc[i, "REF"],
            snp_info.loc[i, "ALT"],
            str(snp_info.loc[i, "QUAL"]),
            "PASS",
            info_field,
            "GT:DP:AD"
        ] + sample_fields)
        
        f.write(line + "\n")
        
        if (i + 1) % 200 == 0:
            print(f"  Written {i + 1}/{N_TOTAL_SNPS} SNPs...")

# =============================================================================
# Write Phenotype/Passport CSV
# =============================================================================
print("Writing phenotype/passport CSV...")

csv_file = "/home/claude/atlantic_giant_pumpkin_diversity_phenotypes.csv"
passport_data.to_csv(csv_file, index=False)

# =============================================================================
# Write SNP Info CSV
# =============================================================================
print("Writing SNP info CSV...")

snp_info_file = "/home/claude/atlantic_giant_pumpkin_diversity_snp_info.csv"

snp_info["Freq_Eastern"] = np.round(ancestral_freqs[:, 0], 3)
snp_info["Freq_Midwest"] = np.round(ancestral_freqs[:, 1], 3)
snp_info["Freq_Western"] = np.round(ancestral_freqs[:, 2], 3)

snp_info.to_csv(snp_info_file, index=False)

# =============================================================================
# Summary Statistics
# =============================================================================
print("\n" + "=" * 70)
print("DIVERSITY PANEL SUMMARY (v3 - Realistic QTL with LD Structure)")
print("=" * 70)

print(f"\nTotal samples: {N_SAMPLES}")
print("\nSamples by Region:")
for region in ["Eastern", "Midwest", "Western"]:
    n = sum(r == region for r in region_assignments)
    print(f"  {region}: {n} samples")

print(f"\nTotal SNPs: {N_TOTAL_SNPS}")
print(f"  High-quality SNPs: {sum(q == 'good' for q in quality_categories)}")
print(f"  Low MAF SNPs: {sum(q == 'low_maf' for q in quality_categories)}")
print(f"  High missing SNPs: {sum(q == 'high_missing' for q in quality_categories)}")

print(f"\n{'='*70}")
print("QTL REGION DETAILS")
print("="*70)
print(f"\nChromosome: {qtl_chrom}")
print(f"\nCausal SNP (the TRUE QTL):")
causal_pos = snp_info.loc[qtl_causal_index, 'POS']
print(f"  {snp_info.loc[qtl_causal_index, 'ID']} at {causal_pos:,} bp")
print(f"  Effect: {QTL_EFFECT_SIZE} lbs per alt allele")

print(f"\nTag SNPs (significant due to LD, NOT causal):")
for tag_idx in qtl_tag_indices:
    tag_pos = snp_info.loc[tag_idx, 'POS']
    distance = abs(tag_pos - causal_pos)
    r2 = calculate_r2(geno_matrix[qtl_causal_index, :], geno_matrix[tag_idx, :])
    print(f"  {snp_info.loc[tag_idx, 'ID']} at {tag_pos:,} bp")
    print(f"    Distance from causal: {distance/1000:.1f} kb")
    print(f"    LD with causal (r²): {r2:.3f}")

print(f"\nTotal QTL region span: {max(snp_info.loc[all_qtl_indices, 'POS']) - min(snp_info.loc[all_qtl_indices, 'POS']):,} bp")

print(f"\n{'='*70}")
print("EXPECTED GWAS RESULTS")
print("="*70)
print(f"""
1. GWAS will show 3 significant SNPs on {qtl_chrom}:
   - All within ~1.5 Mb of each other
   - All appear significant due to association with trait

2. After LD pruning/clumping:
   - Only 1 signal remains (the QTL region)
   - The causal SNP OR a tag SNP may be the lead SNP
     (depends on which has lowest p-value)

3. This demonstrates:
   - Multiple significant SNPs ≠ multiple QTLs
   - LD filtering identifies independent signals
   - Fine-mapping needed to find causal variant
""")

print(f"\nFruit weight statistics:")
print(f"  Overall: {np.mean(fruit_weight):.1f} ± {np.std(fruit_weight):.1f} lbs")
print(f"  Range: {np.min(fruit_weight):.1f} - {np.max(fruit_weight):.1f} lbs")

print("\nFruit weight by Region:")
for region in ["Eastern", "Midwest", "Western"]:
    weights = [w for w, r in zip(fruit_weight, region_assignments) if r == region]
    print(f"  {region}: {np.mean(weights):.1f} ± {np.std(weights):.1f} lbs")

print("\n" + "=" * 70)
print("OUTPUT FILES")
print("=" * 70)
print(f"VCF file: {vcf_file}")
print(f"Phenotype CSV: {csv_file}")
print(f"SNP info CSV: {snp_info_file}")

print("\nDone!")
