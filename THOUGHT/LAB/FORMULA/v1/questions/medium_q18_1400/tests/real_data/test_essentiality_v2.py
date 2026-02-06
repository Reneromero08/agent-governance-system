#!/usr/bin/env python3
"""
Test R vs Gene Essentiality using REAL DepMap data.
Version 2: Uses comprehensive Affymetrix HG-U133 probe-to-gene mapping.
"""

import json
import math
from pathlib import Path

CACHE_DIR = Path(__file__).parent / 'cache'

# Comprehensive HG-U133 Plus 2.0 probe to gene symbol mapping
# Source: Affymetrix annotation files (curated subset of ~500 probes)
PROBE_TO_GENE = {
    '1007_s_at': 'DDR1', '1053_at': 'RFC2', '117_at': 'HSPA6', '121_at': 'PAX8',
    '1255_g_at': 'GUCA1A', '1294_at': 'UBA7', '1316_at': 'THRA', '1320_at': 'PTPN21',
    '1405_i_at': 'CCL5', '1431_at': 'CYP2E1', '1438_at': 'EPHB3', '1487_at': 'ESRRA',
    '200000_s_at': 'PRPF18', '200001_at': 'CAPNS1', '200002_at': 'RPL35',
    '200003_s_at': 'RPL28', '200004_at': 'EIF4G2', '200005_at': 'EIF3D',
    '200006_at': 'PARK7', '200007_at': 'SRP14', '200008_s_at': 'GDI2',
    '200010_at': 'RPL11', '200011_s_at': 'TTC3', '200012_x_at': 'RPL21',
    '200013_at': 'RPL24', '200014_s_at': 'HNRNPC', '200016_x_at': 'RPS15A',
    '200017_at': 'RPL10A', '200018_at': 'RPS13', '200019_s_at': 'FAU',
    '200020_at': 'TARDBP', '200021_at': 'USP33', '200022_at': 'HDLBP',
    '200023_s_at': 'EIF3A', '200024_at': 'PUM1', '200025_s_at': 'RPL27A',
    '200026_at': 'RPL34', '200027_at': 'NACA', '200029_at': 'RPL19',
    '200031_s_at': 'SET', '200032_s_at': 'RPL9', '200033_at': 'DDX5',
    '200034_s_at': 'RPL6', '200036_s_at': 'RPL10', '200037_s_at': 'ACTR2',
    '200038_s_at': 'RPL17', '200039_s_at': 'PSMB2', '200040_at': 'KHSRP',
    '200042_at': 'C14orf2', '200043_at': 'ERH', '200045_at': 'ABCF1',
    '200046_at': 'DAD1', '200047_s_at': 'YY1', '200048_s_at': 'ILK',
    '200049_at': 'TMED2', '200050_at': 'ZDHHC5', '200051_at': 'SART1',
    '200052_s_at': 'ILF2', '200053_at': 'SPCS2', '200054_at': 'ZNF131',
    '200055_at': 'TAF10', '200056_s_at': 'C1D', '200057_s_at': 'NONO',
    '200058_s_at': 'NONO', '200059_s_at': 'RHOA', '200060_s_at': 'RHOA',
    '200061_s_at': 'RPS24', '200062_s_at': 'RPL14', '200063_s_at': 'NPM1',
    '200064_at': 'HSP90AB1', '200065_s_at': 'ARPC1B', '200066_at': 'IK',
    '200067_x_at': 'SNX3', '200068_s_at': 'RHEB', '200069_at': 'SNRPD1',
    '200070_at': 'SDHC', '200071_at': 'SERBP1', '200072_s_at': 'ATP5F1B',
    '200073_s_at': 'SURF4', '200074_s_at': 'RPN1', '200075_s_at': 'GUK1',
    '200076_s_at': 'EEF1A1', '200077_s_at': 'OAZ1', '200078_s_at': 'ATP6V0B',
    '200079_s_at': 'KARS1', '200080_s_at': 'RPS5', '200081_s_at': 'RPS6',
    '200082_s_at': 'RPS7', '200083_at': 'USP22', '200084_at': 'SPEN',
    '200085_s_at': 'TMBIM6', '200086_s_at': 'COX4I1', '200087_s_at': 'TMX1',
    '200088_x_at': 'CLTC', '200089_s_at': 'RPL4', '200090_at': 'RUVBL2',
    '200091_s_at': 'RPS25', '200092_s_at': 'RPL32', '200093_s_at': 'HINT1',
    '200094_s_at': 'EEF2', '200095_x_at': 'RPL36AL', '200096_s_at': 'ATP6V0E1',
    '200685_at': 'ACTB', '200738_s_at': 'PGK1', '200806_s_at': 'HSPD1',
    '200807_s_at': 'HSPD1', '200823_x_at': 'RPL29', '200824_at': 'GSTP1',
    '200825_s_at': 'HYOU1', '200853_at': 'RPS2', '200854_at': 'RPS11',
    '200869_at': 'RPL18A', '200870_at': 'HNRNPA2B1', '200879_s_at': 'PSMD14',
    '200881_s_at': 'DNAJA1', '200882_s_at': 'PSMD14', '200897_s_at': 'PDCD10',
    '200904_at': 'EIF3E', '200905_x_at': 'CDC42', '200906_s_at': 'PALLD',
    '200911_s_at': 'TALDO1', '200924_s_at': 'SLC3A2', '200931_s_at': 'VCL',
    '200933_x_at': 'RPS4X', '200934_at': 'DEK', '200937_s_at': 'RPL5',
    '200938_s_at': 'RPL5', '200953_s_at': 'CCND2', '200959_at': 'FUS',
    '200965_s_at': 'ABLIM1', '200966_x_at': 'ALDOA', '200989_at': 'HIF1A',
    '201091_s_at': 'CBX3', '201125_s_at': 'ANXA2', '201170_s_at': 'BHLHE40',
    '201195_s_at': 'SLC7A5', '201204_s_at': 'RRBP1', '201231_s_at': 'ENO1',
    '201250_s_at': 'SLC2A1', '201272_at': 'AKR1B1', '201283_s_at': 'KRAS',
    '201291_s_at': 'TOP2A', '201292_at': 'TOP2A', '201321_s_at': 'SMARCC2',
    '201389_at': 'ITGA5', '201464_x_at': 'JUN', '201473_at': 'JUNB',
    '201502_s_at': 'NFKBIA', '201531_at': 'ZFP36', '201565_s_at': 'ID2',
    '201566_x_at': 'ID2', '201591_s_at': 'NRAS', '201626_at': 'INSIG1',
    '201627_s_at': 'INSIG1', '201667_at': 'GJA1', '201693_s_at': 'EGR1',
    '201694_s_at': 'EGR1', '201739_at': 'SGK1', '201746_at': 'TP53',
    '201790_s_at': 'DHCR7', '201795_at': 'LBR', '201809_s_at': 'ENG',
    '201830_s_at': 'NET1', '201884_at': 'CEACAM5', '201890_at': 'RRM2',
    '201939_at': 'PLK2', '201983_s_at': 'EGFR', '202014_at': 'PPP1R15A',
    '202052_s_at': 'RAI14', '202086_at': 'MX1', '202107_s_at': 'MCM2',
    '202209_at': 'LSM3', '202270_at': 'GBP1', '202284_s_at': 'CDKN1A',
    '202291_s_at': 'MGP', '202338_at': 'TK1', '202350_s_at': 'MATN2',
    '202388_at': 'RGS2', '202400_s_at': 'SOX15', '202404_s_at': 'CDK2',
    '202431_s_at': 'MYC', '202464_s_at': 'PFKFB3', '202499_s_at': 'SLC2A3',
    '202503_s_at': 'KIAA0101', '202531_at': 'IRF1', '202580_x_at': 'FOXM1',
    '202581_at': 'HSPA1A', '202589_at': 'TYMS', '202628_s_at': 'SERPINE1',
    '202672_s_at': 'ATF3', '202687_s_at': 'TNFSF10', '202704_at': 'TOB1',
    '202770_s_at': 'CCNG2', '202803_s_at': 'ITGB2', '202859_x_at': 'IL8',
    '202897_at': 'PTGS1', '202917_s_at': 'S100A8', '202936_s_at': 'SOX9',
    '203001_s_at': 'STMN2', '203085_s_at': 'TGFB1', '203167_at': 'TIMP2',
    '203214_x_at': 'CDK1', '203276_at': 'LMNB1', '203325_s_at': 'COL5A1',
    '203340_s_at': 'SLC25A36', '203418_at': 'CCNA1', '203438_at': 'STC2',
    '203476_at': 'TPBG', '203561_at': 'FCGR2A', '203562_at': 'FEZ1',
    '203574_at': 'NFIL3', '203576_at': 'BRDT', '203665_at': 'HMOX1',
    '203725_at': 'GADD45A', '203752_s_at': 'JUND', '203753_at': 'TCF4',
    '203767_s_at': 'STS', '203819_s_at': 'IGF2R', '203868_s_at': 'VCAM1',
    '203936_s_at': 'MMP9', '203948_s_at': 'MPO', '203963_at': 'CA12',
    '204041_at': 'MAOB', '204051_s_at': 'SFRP4', '204103_at': 'CCL4',
    '204131_s_at': 'FOXO3', '204214_s_at': 'RAD51', '204285_s_at': 'PMAIP1',
    '204318_s_at': 'GTSE1', '204320_at': 'NR2C2', '204326_x_at': 'MT1X',
    '204388_s_at': 'MAOA', '204420_at': 'FOSL1', '204439_at': 'IFI44L',
    '204457_s_at': 'GAS1', '204490_s_at': 'CD44', '204508_s_at': 'CA12',
    '204531_s_at': 'BRCA1', '204580_at': 'MMP12', '204614_at': 'SERPINB2',
    '204655_at': 'CCL5', '204670_x_at': 'HLA-DRB1', '204748_at': 'PTGS2',
    '204855_at': 'SERPINB5', '204860_s_at': 'NACC1', '204897_at': 'PTGER4',
    '204969_s_at': 'RDH10', '204971_at': 'CSTA', '205009_at': 'TFF1',
    '205027_s_at': 'MAP3K8', '205041_s_at': 'ORM1', '205067_at': 'IL1B',
    '205114_s_at': 'CCL3', '205205_at': 'RELB', '205239_at': 'AREG',
    '205289_at': 'BMP2', '205476_at': 'CCL20', '205569_at': 'LAMP3',
    '205653_at': 'CTSG', '205767_at': 'EREG', '205780_at': 'BIK',
    '205798_at': 'IL7R', '205890_s_at': 'UBD', '205924_at': 'RAB3B',
    '206025_s_at': 'TNFAIP6', '206026_s_at': 'TNFAIP6', '206115_at': 'EGR3',
    '206157_at': 'PTX3', '206172_at': 'IL13RA2', '206204_at': 'GRB14',
    '206211_at': 'SELE', '206295_at': 'IL18', '206332_s_at': 'IFI16',
    '206359_at': 'SOCS3', '206385_s_at': 'ANK3', '206432_at': 'HAS2',
    '206461_x_at': 'MT1H', '206513_at': 'AIM2', '206561_s_at': 'AKR1B10',
    '206569_at': 'IL24', '206584_at': 'LY96', '206622_at': 'TLR4',
    '206666_at': 'GZMK', '206710_s_at': 'EPB41L3', '206785_s_at': 'KLRC1',
    '206932_at': 'CH25H', '206941_x_at': 'SEMA3E', '207001_x_at': 'DTX1',
    '207039_at': 'CDKN2A', '207113_s_at': 'TNF', '207165_at': 'HMMR',
    '207574_s_at': 'GADD45B', '207608_x_at': 'ETV4', '207826_s_at': 'ID3',
    '207850_at': 'CXCL3', '207900_at': 'CCNB2', '207978_s_at': 'NR4A3',
    '208018_s_at': 'HCK', '208075_s_at': 'CCL7', '208078_s_at': 'SNF8',
    '208079_s_at': 'AURKA', '208212_s_at': 'STRAP', '208370_s_at': 'RELA',
    '208451_s_at': 'C4BPA', '208536_s_at': 'ADRBK1', '208614_s_at': 'FLNB',
    '208656_s_at': 'MT1A', '208659_at': 'TNFRSF25', '208664_s_at': 'PTTG1',
    '208711_s_at': 'CCND1', '208712_at': 'CCND1', '208763_s_at': 'TSC22D3',
    '208774_at': 'PCSK1', '208789_at': 'PTRF', '208891_at': 'DUSP6',
    '208892_s_at': 'DUSP6', '208937_s_at': 'ID1', '208944_at': 'TGFBR2',
    '208966_x_at': 'IFI16', '208991_at': 'STAT3', '209016_s_at': 'KRT7',
    '209071_s_at': 'RGS5', '209101_at': 'CTGF', '209116_x_at': 'HBB',
    '209189_at': 'FOS', '209211_at': 'KLF5', '209267_s_at': 'SLC39A8',
    '209270_at': 'LAMB3', '209283_at': 'CRYAB', '209304_x_at': 'GADD45B',
    '209305_s_at': 'GADD45B', '209335_at': 'DCN', '209369_at': 'ANXA3',
    '209397_at': 'MMP7', '209398_at': 'MMP7', '209459_s_at': 'ABAT',
    '209460_at': 'ABAT', '209541_at': 'IGF1', '209604_s_at': 'GATA3',
    '209706_at': 'NKX3-1', '209714_s_at': 'CDKN3', '209773_s_at': 'RRM2',
    '209774_x_at': 'CXCL2', '209795_at': 'CD69', '209831_x_at': 'DNASE2',
    '209875_s_at': 'SPP1', '209921_at': 'SLC7A11', '209949_at': 'NCF2',
    '209969_s_at': 'STAT1', '210001_s_at': 'SOCS1', '210029_at': 'IDO1',
    '210052_s_at': 'TPX2', '210095_s_at': 'IGFBP3', '210163_at': 'CXCL11',
    '210164_at': 'GZMB', '210229_s_at': 'CSF2', '210495_x_at': 'FN1',
    '210512_s_at': 'VEGFA', '210538_s_at': 'BIRC3', '210559_s_at': 'CDC2',
    '210664_s_at': 'TFPI', '210797_s_at': 'OASL', '210845_s_at': 'PLAUR',
    '210900_x_at': 'RPL37', '210986_s_at': 'TPM1', '211300_s_at': 'TP53',
    '211506_s_at': 'IL8', '211518_s_at': 'BID', '211519_s_at': 'KRT18',
    '211527_x_at': 'VEGFA', '211663_x_at': 'PTGDS', '211702_s_at': 'FN1',
    '211712_s_at': 'ANXA7', '211716_x_at': 'HMGA2', '211750_x_at': 'FN1',
    '211821_x_at': 'CST3', '211959_at': 'IGFBP5', '211968_s_at': 'HSP90AA1',
    '211989_at': 'HER3', '212014_x_at': 'CD44', '212063_at': 'CD44',
    '212171_x_at': 'VEGFA', '212185_x_at': 'MT2A', '212239_at': 'PIK3R1',
    '212281_s_at': 'MAC30', '212285_s_at': 'AGRN', '212354_at': 'SULF1',
    '212501_at': 'CEBPB', '212543_at': 'AIM1', '212591_at': 'SEC22B',
    '212592_at': 'IGJ', '212671_s_at': 'HLA-DQA1', '212680_x_at': 'PPP1R14B',
    '212706_at': 'RASA1', '212724_at': 'RND3', '212730_at': 'DMD',
    '212771_at': 'THRSP', '212820_at': 'DMXL2', '212859_x_at': 'MT1E',
    '212942_s_at': 'KIAA1199', '213006_at': 'CEBPD', '213110_s_at': 'COL4A5',
    '213258_at': 'TFAP2C', '213274_s_at': 'FSTL3', '213338_at': 'TMEM158',
    '213418_at': 'HSPA6', '213524_s_at': 'G0S2', '213562_s_at': 'SQLE',
    '213629_x_at': 'MT1F', '213797_at': 'RSAD2', '213831_at': 'HLA-DQA1',
    '213867_x_at': 'OST48', '213921_at': 'SSR2', '213938_at': 'RBMS3',
    '214039_s_at': 'LAPTM4B', '214059_at': 'IFI44', '214079_at': 'DHRS2',
    '214321_at': 'NOXO1', '214438_at': 'FKBP5', '214453_s_at': 'IFI44',
    '214575_s_at': 'MT1H', '214696_at': 'GPM6A', '214702_at': 'FN1',
    '214710_s_at': 'CCNB1', '214767_s_at': 'CECR1', '214768_x_at': 'CECR1',
    '214974_x_at': 'CXCL5', '215034_s_at': 'TM4SF1', '215078_at': 'IRGQ',
    '215071_s_at': 'HIST1H2AC', '215118_s_at': 'ABCC1', '215223_s_at': 'SOD2',
    '215306_at': 'GNLY', '215446_s_at': 'LOX', '215485_s_at': 'DEFA1',
    '215501_s_at': 'DEFA3', '215646_s_at': 'VCL', '215674_at': 'CRYAB',
    '215785_s_at': 'CYFIP2', '215867_x_at': 'CA2', '215913_s_at': 'LANCL1',
    '215978_x_at': 'FN1', '215990_s_at': 'IGLL5', '216041_x_at': 'GRM7',
    '216236_s_at': 'SLC2A14', '216598_s_at': 'CCL2', '216834_at': 'RGS1',
    '217022_s_at': 'IGH@', '217028_at': 'CXCR4', '217165_x_at': 'MT1F',
    '217173_s_at': 'LDLR', '217234_s_at': 'VIL1', '217335_x_at': 'HBA1',
    '217388_s_at': 'KYNU', '217398_x_at': 'GAPDH', '217414_x_at': 'HBA2',
    '217428_s_at': 'COL10A1', '217478_s_at': 'HLA-DMA', '217502_at': 'IRF9',
    '217546_at': 'MT1M', '217679_x_at': 'PTMA', '217728_at': 'S100A6',
    '217738_at': 'PBEF1', '217800_s_at': 'NDRG1', '217852_s_at': 'ACP5',
    '217853_at': 'TNS3', '217897_at': 'FXYD6', '217967_s_at': 'MYL9',
    '217975_at': 'SUZ12', '217979_at': 'TSPAN13', '218009_s_at': 'PRC1',
    '218074_at': 'FA2H', '218086_at': 'NPDC1', '218145_at': 'TRIB3',
    '218346_s_at': 'SESN2', '218469_at': 'GREM1', '218541_s_at': 'C8orf4',
    '218585_s_at': 'DTL', '218644_at': 'PLEK2', '218718_at': 'PDGFC',
    '218729_at': 'LXN', '218854_at': 'DSE', '218888_s_at': 'NETO2',
    '218918_at': 'MAN1C1', '218966_at': 'MYO5C', '218980_at': 'FHOD3',
    '218983_at': 'C1RL', '218995_s_at': 'EDN1', '219014_at': 'PLAC8',
    '219211_at': 'USP18', '219263_at': 'RNF144B', '219304_s_at': 'PDGFD',
    '219423_x_at': 'TNFRSF21', '219434_at': 'TREM1', '219607_s_at': 'MS4A4A',
    '219682_s_at': 'TBX3', '219724_s_at': 'RFFL', '219737_s_at': 'PCDH9',
    '219795_at': 'SLC6A14', '219825_at': 'CYP26A1', '219871_at': 'SYT17',
    '219890_at': 'CLEC5A', '219922_s_at': 'LTBP3', '219989_at': 'MNDA',
    '220026_at': 'CLEC12A', '220088_at': 'C5AR1', '220146_at': 'TLR7',
    '220322_at': 'TMEM88', '220560_at': 'MARCH3', '220621_at': 'KIF26B',
    '220651_s_at': 'MCM10', '220858_at': 'ANGPTL4', '221009_s_at': 'ANGPTL4',
    '221019_s_at': 'COLEC12', '221234_s_at': 'BACH2', '221476_s_at': 'RPL15',
    '221491_x_at': 'HLA-DRB5', '221521_s_at': 'PTTG1', '221577_x_at': 'GDF15',
    '221667_s_at': 'HSPB8', '221698_s_at': 'CLDN7', '221755_at': 'PDSS1',
    '221766_s_at': 'FAM46A', '221841_s_at': 'KLF4', '221900_at': 'COL8A2',
    '221958_s_at': 'CSNK1G1', '222088_s_at': 'SLC2A3', '222162_s_at': 'ADAMTS1',
    '222170_at': 'ZNF511', '222250_s_at': 'MASP1', '222446_s_at': 'ABCC6',
    '222698_s_at': 'MAP3K8', '222853_at': 'GCNT3', '222976_s_at': 'KLHL21',
    '223229_at': 'UBE2G2', '223333_s_at': 'ANGPTL4', '223394_at': 'RAB40B',
    '223454_at': 'TMEM109', '223467_at': 'TMEM63A', '223484_at': 'C15orf48',
    '223500_at': 'TRIM22', '223532_at': 'CNIH4', '223592_at': 'ANXA11',
    '223679_at': 'RHBDF2', '223721_s_at': 'ANXA11', '223775_at': 'CKAP2',
    '223817_at': 'DUSP16', '224008_s_at': 'ARHGAP29', '224204_x_at': 'ETV6',
    '224560_at': 'TGIF2', '224825_at': 'EIF2B5', '225081_s_at': 'CHST15',
    '225082_at': 'CHST15', '225167_at': 'SLC35B2', '225404_at': 'TOX',
    '225543_at': 'RBMXL1', '225577_at': 'SYTL2', '225601_at': 'RAB3D',
    '225681_at': 'CTHRC1', '225716_at': 'LRP8', '225798_at': 'LGALS3BP',
    '225870_at': 'PARP14', '225930_at': 'CRYAB', '226064_s_at': 'DYDC1',
    '226110_at': 'SLFN5', '226147_s_at': 'PIGR', '226226_at': 'ABI3BP',
    '226322_at': 'CLDN23', '226360_at': 'KIFC1', '226597_at': 'DSEL',
    '226702_at': 'NUSAP1', '226924_at': 'ZBED2', '226950_at': 'NEXN',
    '227006_at': 'KLHL24', '227140_at': 'EGFL6', '227404_s_at': 'EGR1',
    '227697_at': 'SLC30A3', '227730_at': 'ADTRP', '227817_at': 'ESCO2',
    '228067_at': 'C7orf10', '228152_s_at': 'THSD7A', '228499_at': 'PLXNC1',
    '228531_at': 'P2RY14', '228584_at': 'NLRP3', '228638_at': 'PLCH1',
    '228769_at': 'CDK6', '228792_at': 'ANKRD37', '229043_at': 'MIR503HG',
    '229331_at': 'NFKBID', '229450_at': 'ARMCX2', '229461_x_at': 'TRAC',
    '229498_at': 'HIST2H2BF', '229530_at': 'ACSM3', '229696_at': 'SYTL3',
    '230174_at': 'RASGEF1B', '230236_at': 'LIN7A', '230493_at': 'GPR132',
    '230620_at': 'BAALC', '230748_at': 'SLC43A3', '230795_at': 'CYP7B1',
    '230866_at': 'FCRL6', '230988_at': 'PKIA', '231031_at': 'FAM198B',
    '231041_at': 'IL21R', '231107_at': 'ITGB7', '231202_at': 'FAM107A',
    '231577_s_at': 'GBP1', '231724_at': 'GCNT4', '231775_at': 'HCST',
    '231818_at': 'TPSD1', '231885_at': 'PLXDC2', '231942_at': 'SDPR',
    '232060_at': 'ANXA8L1', '232121_at': 'ADAMTS2', '232164_s_at': 'IRAK3',
    '232165_at': 'IRAK3', '232224_at': 'FAM26F', '232306_at': 'TMEM176A',
    '232340_at': 'ROPN1L', '232344_at': 'ABCC4', '232431_at': 'MCOLN2',
    '232617_at': 'PCSK5', '232779_at': 'RASSF4', '232894_at': 'LRRC25',
    '233018_at': 'SLC26A10', '233073_at': 'KIT', '233323_at': 'KCNN4',
    '233413_at': 'FBN2', '233442_at': 'PLXNA4', '233488_at': 'MARCH1',
    '234071_x_at': 'KLHL22', '234170_at': 'LAX1', '234236_at': 'IFITM3',
    '235003_at': 'KIAA1522', '235107_at': 'NR4A1', '235151_at': 'GBP5',
    '235177_at': 'CDKN2B', '235271_at': 'ISL1', '235276_at': 'KDR',
    '235372_at': 'SPATA18', '235404_at': 'DKK1', '235415_at': 'NKAIN4',
    '235428_at': 'VNN3', '235477_at': 'PADI2', '235535_x_at': 'GGT5',
    '235673_at': 'EIF5A2', '235722_at': 'ARRDC3', '235812_at': 'NRG1',
    '235816_at': 'RASL12', '235887_at': 'SULF2', '235967_at': 'CYBB',
    '236153_at': 'ANXA8', '236165_at': 'ZFR2', '236222_at': 'ARRDC4',
    '236295_at': 'PGLYRP1', '236341_at': 'STARD13', '236455_at': 'ABAT',
    '236538_at': 'PREX1', '236589_at': 'HEY2', '236638_at': 'NLRP2',
    '236669_at': 'GALNTL2', '236735_at': 'SH3TC1', '237075_at': 'WNT5A',
    '237356_at': 'MUC17', '237469_at': 'TRPC4', '237508_at': 'SEMA3C',
    '237511_at': 'KIF7', '237512_at': 'FOS', '237591_at': 'CTNNA2',
    '238689_at': 'SLC5A6', '238795_at': 'DNAJA4', '238825_at': 'EBF4',
    '238851_at': 'SLAMF8', '238966_at': 'MNS1', '239046_at': 'BEST3',
    '239047_at': 'CPXM1', '239168_at': 'DUSP18', '239211_at': 'SLC25A45',
    '239231_at': 'CSGALNACT1', '239380_at': 'SCN5A', '239381_at': 'ATP13A4',
    '239382_at': 'TMEM100', '239629_at': 'ADORA2A', '239667_at': 'DEPTOR',
    '239836_at': 'SLC35E3', '239895_at': 'ADAMTS14', '239973_at': 'TIAM2',
    '240103_at': 'GBP4', '240234_at': 'FAT3', '240260_at': 'RAB15',
    '240407_at': 'HTRA4', '240433_at': 'TMEM161B', '240490_at': 'MFSD4',
    '240766_at': 'SLC35F3', '240821_at': 'NUP210', '240839_at': 'LOC23117',
    '240854_at': 'XKR4', '240868_at': 'CASS4', '240984_at': 'VWCE',
    '241067_at': 'PIK3CG', '241209_at': 'WDR64', '241230_at': 'LPHN3',
    '241410_at': 'KIAA1045', '241480_at': 'ADAMTS5', '241499_at': 'PLBD1',
    '241510_at': 'KIAA1324L', '241527_at': 'FAM81A', '241592_at': 'DPP6',
    '241648_at': 'NTRK3', '241700_at': 'SEMA6A', '241750_at': 'DUSP15',
    '241765_at': 'PDGFRB', '241827_at': 'TMEM132C', '241829_at': 'TRAF3IP3',
    '242014_at': 'SGIP1', '242042_s_at': 'FAM5C', '242069_at': 'RIMS2',
    '242146_at': 'PADI4', '242234_at': 'LOC648149', '242287_at': 'PLCXD3',
    '242322_at': 'GPR126', '242425_at': 'RAB6B', '242481_at': 'ADAM23',
    '242517_at': 'MYCT1', '242560_at': 'AADACL1', '242611_at': 'POU3F1',
    '242625_at': 'RAI2', '242689_at': 'PLIN2', '242837_at': 'LOC100134713',
    '242857_at': 'LOC399744', '242904_at': 'TNFSF15', '242909_at': 'ITGBL1',
    '242954_at': 'CTHRC1', '243077_at': 'CCDC140', '243135_at': 'KCTD4',
    '243214_at': 'ENPEP', '243228_at': 'LOC100505633', '243296_at': 'C1orf182',
    '243362_s_at': 'SAMD9L', '243509_at': 'FAM5B', '243510_at': 'RAPH1',
    '243609_s_at': 'NFKBIZ', '243627_at': 'COL22A1', '243700_at': 'PLA2G4C',
    '243779_at': 'ADCY2', '243817_at': 'PRKAR1B', '243818_at': 'ZC3H12A',
    '243913_at': 'MOSPD1', '243937_at': 'PITPNM3', '244014_at': 'MYBPC2',
    '244534_at': 'TJP3', '32128_at': 'BCAT1', '33197_at': 'ABCD3',
    '34764_at': 'PLAU', '35820_at': 'GMFB', '36711_at': 'MAFF',
    '37152_at': 'PLSCR1', '37892_at': 'COL8A1', '38241_at': 'HYAL2',
    '38269_at': 'CSTA', '38449_s_at': 'ADAM15', '38487_at': 'STAC',
    '39580_at': 'NQO2', '39705_at': 'HOXA5', '40016_g_at': 'CUGBP2',
    '40093_at': 'IGHG1', '40400_at': 'C1QB', '40420_at': 'CDKN1C',
    '40446_at': 'IGFBP2', '40489_at': 'ADORA1', '40687_at': 'BMP4',
    '40704_at': 'RGS2', '41191_at': 'SAA4', '41386_i_at': 'RBP4',
    '41387_r_at': 'RBP4', '41469_at': 'ADH4', '41471_at': 'MBL2',
    '41644_at': 'SERPINA3', '44072_at': 'FOLH1', '44790_s_at': 'ANGPTL2',
    '59644_at': 'DUSP5', '64438_at': 'IRAK2', '65588_at': 'CYP1B1',
    '65630_at': 'TNFAIP3', '65633_at': 'BCL10', '65717_at': 'CBLB',
    '66041_at': 'PRSS23', '87100_at': 'BIRC5', '91816_f_at': 'OAS1',
    '91920_at': 'DDR2', '92020_s_at': 'USP46', '92428_at': 'DEPDC1B',
}


def load_data():
    """Load both datasets."""
    with open(CACHE_DIR / 'depmap_essentiality.json', 'r') as f:
        depmap = json.load(f)['genes']
    with open(CACHE_DIR / 'gene_expression_sample.json', 'r') as f:
        expression = json.load(f)['genes']
    return depmap, expression


def map_probes_to_genes(expression):
    """Map probe IDs to gene symbols and get R values."""
    gene_r = {}
    for probe_key, data in expression.items():
        probe_id = probe_key.split(':')[1] if ':' in probe_key else probe_key
        if probe_id in PROBE_TO_GENE:
            gene = PROBE_TO_GENE[probe_id]
            if gene not in gene_r or data['R'] > gene_r[gene]['R']:
                gene_r[gene] = {'R': data['R'], 'mean_expr': data['mean_expr'],
                               'std_expr': data['std_expr'], 'probe_id': probe_id}
    return gene_r


def pearson_correlation(x, y):
    """Compute Pearson correlation."""
    n = len(x)
    if n < 3: return 0.0
    mx, my = sum(x)/n, sum(y)/n
    cov = sum((x[i]-mx)*(y[i]-my) for i in range(n))/n
    sx = math.sqrt(sum((xi-mx)**2 for xi in x)/n)
    sy = math.sqrt(sum((yi-my)**2 for yi in y)/n)
    return cov/(sx*sy) if sx > 1e-10 and sy > 1e-10 else 0.0


def spearman_correlation(x, y):
    """Compute Spearman rank correlation."""
    def rank(v):
        s = sorted(range(len(v)), key=lambda i: v[i])
        r = [0]*len(v)
        for i, idx in enumerate(s, 1): r[idx] = i
        return r
    return pearson_correlation(rank(x), rank(y))


def compute_auc(labels, scores):
    """Compute AUC for binary classification."""
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0: return 0.5
    pairs = sorted(zip(scores, labels), key=lambda x: -x[0])
    sum_ranks = sum(i+1 for i, (_, l) in enumerate(pairs) if l)
    u = sum_ranks - n_pos*(n_pos+1)/2
    return u/(n_pos*n_neg)


def main():
    print("="*70)
    print("Q18: R vs Gene Essentiality Test (Real DepMap Data)")
    print("="*70)

    depmap, expression = load_data()
    print(f"\nDepMap genes: {len(depmap)}")
    print(f"Expression probes: {len(expression)}")

    gene_r = map_probes_to_genes(expression)
    print(f"Mapped to genes: {len(gene_r)}")

    # Match genes
    matched = []
    for gene, r_data in gene_r.items():
        if gene in depmap:
            matched.append({
                'gene': gene, 'R': r_data['R'],
                'mean_effect': depmap[gene]['mean_effect'],
                'essential': depmap[gene]['essential']
            })

    print(f"Matched genes: {len(matched)}")

    if len(matched) < 10:
        print("ERROR: Too few matched genes")
        return

    # Extract data
    r_vals = [m['R'] for m in matched]
    effects = [m['mean_effect'] for m in matched]
    essentials = [m['essential'] for m in matched]

    # Compute statistics
    pearson_r = pearson_correlation(r_vals, effects)
    spearman_r = spearman_correlation(r_vals, effects)
    auc = compute_auc(essentials, r_vals)

    ess_r = [m['R'] for m in matched if m['essential']]
    non_r = [m['R'] for m in matched if not m['essential']]

    mean_r_ess = sum(ess_r)/len(ess_r) if ess_r else 0
    mean_r_non = sum(non_r)/len(non_r) if non_r else 0

    print(f"\n{'='*70}")
    print("RESULTS")
    print("="*70)
    print(f"Matched genes: {len(matched)}")
    print(f"  Essential: {len(ess_r)}")
    print(f"  Non-essential: {len(non_r)}")
    print(f"\nCorrelations (R vs CRISPR effect):")
    print(f"  Pearson r: {pearson_r:.4f}")
    print(f"  Spearman r: {spearman_r:.4f}")
    print(f"\nAUC (R predicts essential): {auc:.4f}")
    print(f"\nMean R by essentiality:")
    print(f"  Essential genes: {mean_r_ess:.4f}")
    print(f"  Non-essential genes: {mean_r_non:.4f}")
    print(f"  Difference: {mean_r_ess - mean_r_non:.4f}")

    # Interpretation
    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print("="*70)

    if pearson_r < -0.1:
        corr_interp = "SUPPORTS: Higher R -> more essential (more negative effect)"
    elif pearson_r > 0.1:
        corr_interp = "CONTRADICTS: Higher R -> less essential"
    else:
        corr_interp = "WEAK: No clear R-essentiality relationship"

    if auc > 0.55:
        auc_interp = "SUPPORTS: R has predictive power for essentiality"
    elif auc < 0.45:
        auc_interp = "CONTRADICTS: Low R predicts essentiality"
    else:
        auc_interp = "WEAK: R marginally predicts essentiality"

    if mean_r_ess > mean_r_non and auc > 0.5:
        conclusion = "SUPPORTED: Essential genes have higher R (more consistent expression)"
    elif mean_r_ess < mean_r_non and auc < 0.5:
        conclusion = "REJECTED: Essential genes have lower R"
    else:
        conclusion = "INCONCLUSIVE: Mixed evidence"

    print(f"Correlation: {corr_interp}")
    print(f"AUC: {auc_interp}")
    print(f"\nCONCLUSION: {conclusion}")

    # Save results
    results = {
        'analysis': 'R vs Gene Essentiality',
        'data_sources': {
            'essentiality': 'DepMap CRISPR (17,916 genes)',
            'expression': 'GEO Series Matrix (HG-U133 Plus 2.0)'
        },
        'sample_sizes': {
            'depmap_genes': len(depmap),
            'expression_probes': len(expression),
            'mapped_genes': len(gene_r),
            'matched_genes': len(matched),
            'essential_matched': len(ess_r),
            'nonessential_matched': len(non_r)
        },
        'statistics': {
            'pearson_r': round(pearson_r, 4),
            'spearman_r': round(spearman_r, 4),
            'auc': round(auc, 4),
            'mean_r_essential': round(mean_r_ess, 4),
            'mean_r_nonessential': round(mean_r_non, 4),
            'r_difference': round(mean_r_ess - mean_r_non, 4)
        },
        'interpretation': {
            'correlation': corr_interp,
            'auc': auc_interp,
            'conclusion': conclusion
        },
        'validation': {
            'is_circular': False,
            'reason': 'R from expression consistency, essentiality from CRISPR knockouts'
        },
        'matched_genes_sample': matched[:30]
    }

    output_file = Path(__file__).parent / 'essentiality_test_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == '__main__':
    main()
