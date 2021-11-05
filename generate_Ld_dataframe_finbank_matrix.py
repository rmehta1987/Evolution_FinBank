import glob
import pygwasvcf
import pysam
import os
import subprocess # for running plink
import h5py as hf
import numpy as np
from tqdm import tqdm

# Flag Parser 
from absl import app 
from absl import flags

FLAGS = flags.FLAGS
'''
Mainly to see if data-frame that contains names of summary statitistics traits
already exists 
'''
# Data params
flags.DEFINE_string('dataframe_name', 'fin_biobank_vcf.pkl', 'Dataframe (pkl) File Name')
flags.DEFINE_string('path_to_vcf_files', "/home/ludeep/Desktop/PopGen/FinBank/open_gwas_data_vcf/", 'Path to summary statistics')
flags.DEFINE_integer('num_traits', 1, 'Number of traits to extract data from')
flags.DEFINE_integer('num_snps', 660000, 'Number of SNPS to extract from dataset, 0 for all SNPS')
flags.DEFINE_float('ld_threshold', 10e-8, "Threshold of LD matrix values, anything less than or equal to is 0.")
flags.DEFINE_string('plink_path', "/usr/bin/plink1.9", "Path to Plink")
flags.DEFINE_string('reference_path', "/home/ludeep/Desktop/PopGen/eqtlGen/Reference/1kg.v3.tgz", "Path to Reference File")

flags.register_validator('num_snps',
                         lambda value: value % 22 == 0,
                         message='--num_snps must be divisible by 22 or 0 for all SNPS')
def random_unique_indexes_per_row(A: np.ndarray, N: int=2):
    """Choose random SNPS (column) from each trait (row)
    https://stackoverflow.com/questions/51279464/sampling-unique-column-indexes-for-each-row-of-a-numpy-array

    Args:
        A ([np.ndarray]): [description]
        N (int): [num of columns to choose]. Defaults to 2.

    Returns:
        [type]: [description]
    """
    m,n = A.shape
    return np.random.rand(m,n).argsort(1)[:,:N]

def getRSIDS(path_to_vcf_files: str, num_traits: int,num_snps: int):
    """Get RSIDS in num_traits vcf files limited to num_snps

    Args:
        path_t_vcf_files (str): path to summary statistics in vcf files
        num_traits (int): number of traits to collate in dataset
        num_snps (int): number of SNPS to collate in dataset
    """    

    vcf_files = glob.glob("{}*.gz".format(path_to_vcf_files))
    if len(vcf_files) == num_traits:
        print("Using all traits in dataset")
    else:
        vcf_files_random_choice = np.random.choice(vcf_files,size=num_traits,replace=False)
        print("Random sampled traits from dataset, saved as random_vcf_files.pkl")
        vcf_files = np.copy(vcf_files_random_choice)
        np.save('random_vcf_files.pkl',vcf_files)
        del vcf_files_random_choice
        
    for a_vcf_file in vcf_files:
        with pygwasvcf.GwasVcf(a_vcf_file) as g, pysam.VariantFile(a_vcf_file) as samfile:
            trait_name = g.get_traits()[0]
            trait_rsids_dir = os.path.join(path_to_vcf_files,'{}_rsids'.format(trait_name))
            if not (os.path.isdir(trait_rsids_dir)):
                try:
                    os.mkdir(trait_rsids_dir)
                except OSError:
                    print("Making directory")
            num_variants = int(g.get_metadata()[trait_name]['TotalVariants']) - int(g.get_metadata()[trait_name]['VariantsNotRead'])
            print("Number of variants {} in the trait {}".format(num_variants, trait_name))
            # gets a list of all the contigs from the header (assumes header info has contigs)
            the_contigs = list(samfile.header.contigs) 
            rsIDs = np.empty((len(the_contigs),num_variants), dtype=object) # Hopefully no out of index error to be efficient, also not efficient because unknown length of string
            for i, a_contig in enumerate(tqdm(the_contigs)):
                for ind, variant in enumerate(g.query(contig=a_contig)):
                    rsIDs[i, ind]=pygwasvcf.VariantRecordGwasFuns.get_id(variant, trait_name)
            rsid_listname='{}{}_all_variants_file.pkl'.format(path_to_vcf_files,trait_name)
            # now choose 
            np.save(rsid_listname, rsIDs)
            if num_snps != 0:
                print ("Using a smaller number of varints, {}, in comparison to total number of SNPS, so downsampling SNPs.".format(num_snps))
                newRSIds = np.take(rsIDs, random_unique_indexes_per_row(rsIDs, num_snps))
                rsid_listname='{}{}_{}_variants_file.pkl'.format(path_to_vcf_files,trait_name,num_snps)
                np.save(rsid_listname,newRSIds)
                print("Saved RSIDS of trait {}".format(trait_name))
                del newRSIds
            else:
                raise Exception("Number of variants requested is greater than number of SNPS in dataset, use 0 to request all variants")
            del rsIDs
            
            
           
def generateLD(rsIDs: np.ndarray, ld_threshold: float):
    """Generate an diagonal block matrix consisting of rsIDs, rsIDs on different chromosomes
    are zero, and anything less than threshold will be 0. If shape of rsIDS is too large (X x M),
    M > 1000000 number of SNPS it will require a large memory and hard-drive diskspace
    
    Args:
        rsIDs (np.ndarray): X x M shape chronosomes x num_snps
        ld_threshold (float): Ld threshold anything below will be zero
    """
    #location of plink /usr/bin/plink1.9
    #location of refernce: /home/ludeep/Desktop/PopGen/eqtlGen/Reference
    def execute_command(command):
        print("Execute command: {}".format(command))
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(process.communicate()[0].decode("utf-8"))
    
    
    # Create a LD matrix for every chromosome
    chromosomes, num_snps = rsIDs.shape
    for contig in chromosomes:
        np.savetxt('rsidFil.txt',rsIDs[contig],fmt="%s")
        # Create LD file in table format
        execute_command('{0} --bfile {1} --r2 triangle bin --extract {} --ld-window-r2 {3} --out {1}'.format(path_to_plink, path_to_bfile, ld_threshold, 'ld.out')) 
          
def main(argv):
    
   
    path_to_vcf_files = FLAGS.path_to_vcf_files
    num_traits = FLAGS.num_traits
    num_snps = FLAGS.num_snps
    
    getRSIDS(path_to_vcf_files, num_traits, num_snps)
    #generateLD()


if __name__ == '__main__':
    app.run(main)
    
'''
for variant in g.query(contig="1", start=1, stop=1):
        # print variant-trait P value
        print(pygwasvcf.VariantRecordGwasFuns.get_pval(variant, "trait_name"))
        # print variant-trait SE
        print(pygwasvcf.VariantRecordGwasFuns.get_se(variant, "trait_name"))
        # print variant-trait beta
        print(pygwasvcf.VariantRecordGwasFuns.get_beta(variant, "trait_name"))
        # print variant-trait allele frequency
        print(pygwasvcf.VariantRecordGwasFuns.get_af(variant, "trait_name"))
        # print variant-trait ID
        print(pygwasvcf.VariantRecordGwasFuns.get_id(variant, "trait_name"))
        # create and print ID on-the-fly if missing
        print(pygwasvcf.VariantRecordGwasFuns.get_id(variant, "trait_name", create_if_missing=True))
        # print variant-trait sample size
        print(pygwasvcf.VariantRecordGwasFuns.get_ss(variant, "trait_name"))
        # print variant-trait total sample size from header if per-variant is missing
        print(pygwasvcf.VariantRecordGwasFuns.get_ss(variant, "trait_name", g.get_metadata()))
        # print variant-trait number of cases
        print(pygwasvcf.VariantRecordGwasFuns.get_nc(variant, "trait_name"))
        # print variant-trait total cases from header if per-variant is missing
        print(pygwasvcf.VariantRecordGwasFuns.get_nc(variant, "trait_name", g.get_metadata()))


'''