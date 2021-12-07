import numpy as np
import glob
from tqdm import tqdm

# Need to find unique variants among all the traits

# First get all varints stored as numpy arrays from folder: /project2/jjberg/data/summary_statistics/Fin_BANK/open_gwas_data_vcf

all_variant_files = glob.glob('/project2/jjberg/data/summary_statistics/Fin_BANK/open_gwas_data_vcf/**/*_all_variants_file.npy')
set_file = set()

# for loop to iterate through files
for file in tqdm(all_variant_files):
    temp_file = np.load(file)
    snps = np.where(temp_file.flatten()!=None)[0]
    del temp_file # for memory constraints
    print("Current size of set: {} and number of SNPs in current trait: {}".format(len(set_file),snps.shape[0]))
    if set_file:
        set_file.update(set(snps))
    else:
        set_file = set(snps)
    del snps
   
    
set_array = np.fromiter(set_file, dtype=object)

print("Saving unique RSIDs to numpy array")
# Path to save files
path_to_save_set = '/project2/jjberg/data/summary_statistics/Fin_BANK/open_gwas_data_vcf/unique_rsid_sets.npy'

np.save(path_to_save_set,set_array)

path_to_save_set_txt = '/project2/jjberg/data/summary_statistics/Fin_BANK/open_gwas_data_vcf/unique_rsid__text_sets.txt'
 

print("Saving unique RSIDs to textfile")
np.savetxt(path_to_save_txt, set_array, fmt="%s",delimiter='\n')
