###Calculate pgs distribution for all models using 1kg data.

Modify the memory optimized script (optim_1kg_v_10, see 10_optimize_memory.md), and then apply the following criteria to generate a new script calculate_distribution_pgs_1kg.py. This script will generate reference distribution (and relevant meta data) for all pgs model using 1kg data (if applicable) within 1kg_pgs_reference.

Fix a frozen scoring manifest once, and apply it identically to the panel and
  to every future individual:

  1. Define S ⊆ N by call rate, not by intersection. Keep locus j if it matched
     the model and its call rate across the M panel ≥ ~0.95 (using the union set
     as the candidate pool — your D already gives this for free as
     obs.sum(axis=1)/M). Report |S|/N as the model's coverage; if it's low, the
     model is not deployable on this genotyping platform and should be flagged,
     not quietly scored.
  2. Mean-impute the residual missingness within S, using the panel
     effect-allele frequency: missing dosage → 2·p_j. This keeps E[score]
     unbiased and is what plink2 does by default. This is one extra sparse
     matmul on your existing kernel — you already compute obs; add scores += 
     Wimp @ (1 - obs) where Wimp holds sign_j·w_j·2p_j on the same sparsity
     pattern as Wd. Same cost class as the Wo term you already have.
  3. Freeze p_j with the manifest. This is the part that's easy to get wrong: a
     new individual scored later must be imputed with the reference panel's p_j,
     not a recomputed one, or their score isn't on the reference scale. Store
     per locus: key, effect allele, weight, flip flag, p_j, panel call rate.
  4. Emit per-sample coverage next to every score. n_obs[i,j] = A @ obs where A
     is the 0/1 pattern of Wd — one more matmul, per-model this time, which is
     what DENOM should have been. Then QC on it: flag/drop samples below ~0.99
     call rate over S rather than letting them carry a shrunken score. This is
     the single cheapest addition and it makes every other problem visible.
  5. Standardize last, and stratify. z = (score − μ)/σ from the panel. With an
     AoU-style multi-ancestry M, a single μ, σ is dominated by ancestry-driven
     allele-frequency differences — regress the raw score on the panel's top
     ~4–10 genotype PCs and standardize the residual (or fit μ(PC), σ(PC)), then
     store those coefficients in the manifest too.
  6. Resolve absent vs. no-call before any of this. Ideally score from a joint
     callset or reference-confident representation so 0/0 is explicit. If you
     must consume variant-only per-sample VCFs, you need a per-sample
     callable-region definition: inside callable regions, absent = hom-ref
     (dosage 0, obs=1); outside, absent = missing (obs=0). Otherwise step 2
     imputes loci that were actually observed hom-ref, and flipped variants stay
     wrong.
