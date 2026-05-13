# N1 Exploratory Label-Purity Follow-Up

Model: `sentence-transformers/all-MiniLM-L6-v2`

## ag_news

- Source: ag_news test
- Spearman: E=0.7954, R_simple=0.7315, R_full=0.7240
- AUC pure-vs-not: E=0.9263, R_simple=0.8956, R_full=0.8925
- Delta Spearman E-R_simple: 0.0641 [0.0295, 0.1065]
- Delta AUC E-R_simple: 0.0311 [0.0043, 0.0612]
- Winner by Spearman: E
- Winner by AUC: E

## emotion

- Source: dair-ai/emotion test
- Spearman: E=0.1467, R_simple=0.1075, R_full=0.0989
- AUC pure-vs-not: E=0.5892, R_simple=0.5775, R_full=0.5790
- Delta Spearman E-R_simple: 0.0408 [-0.0592, 0.1400]
- Delta AUC E-R_simple: 0.0104 [-0.0605, 0.0768]
- Winner by Spearman: tie
- Winner by AUC: tie

## sst2

- Source: glue/sst2 validation
- Spearman: E=0.2131, R_simple=0.1822, R_full=0.1732
- AUC pure-vs-not: E=0.6205, R_simple=0.6018, R_full=0.5957
- Delta Spearman E-R_simple: 0.0294 [-0.0664, 0.1298]
- Delta AUC E-R_simple: 0.0194 [-0.0368, 0.0804]
- Winner by Spearman: tie
- Winner by AUC: tie

## snli

- Source: snli validation
- Spearman: E=0.0604, R_simple=0.0707, R_full=0.0684
- AUC pure-vs-not: E=0.5315, R_simple=0.5302, R_full=0.5294
- Delta Spearman E-R_simple: -0.0088 [-0.0586, 0.0437]
- Delta AUC E-R_simple: 0.0007 [-0.0331, 0.0309]
- Winner by Spearman: tie
- Winner by AUC: tie

## mnli

- Source: glue/mnli validation_matched
- Spearman: E=-0.0160, R_simple=-0.0360, R_full=-0.0333
- AUC pure-vs-not: E=0.4932, R_simple=0.4699, R_full=0.4707
- Delta Spearman E-R_simple: 0.0191 [-0.0378, 0.0749]
- Delta AUC E-R_simple: 0.0234 [-0.0097, 0.0568]
- Winner by Spearman: tie
- Winner by AUC: tie
