import glob
import pygwasvcf
import pysam
import os
import subprocess # for running plink
import h5py as hf
import numpy as np
from tqdm import tqdm
import re
import typing
from typing import List
# Flag Parser 
from absl import app 
from absl import flags

FLAGS = flags.FLAGS
'''
Mainly to see if data-frame that contains names of summary statitistics traits
already exists 
'''

temp_path_to_files = '/project2/jjberg/data/summary_statistics/Fin_BANK/open_gwas_data_vcf/'
temp_path_to_reference = '/project2/jjberg/data/1kg/Reference/1kg.v3/EUR/EUR'
temp_path_to_plink='/software/plink-1.90b6.9-el7-x86_64/plink'

# Data params
flags.DEFINE_string('dataframe_name', 'fin_biobank_vcf.pkl', 'Dataframe (pkl) File Name')
flags.DEFINE_string('path_to_vcf_files', temp_path_to_files, 'Path to summary statistics')
flags.DEFINE_integer('num_traits', 1000, 'Number of traits to extract data from')
flags.DEFINE_integer('num_snps', 660000, 'Number of SNPS to extract from dataset, 0 for all SNPS')
flags.DEFINE_float('ld_threshold', 10e-8, "Threshold of LD matrix values, anything less than or equal to is 0.")
flags.DEFINE_string('plink_path',temp_path_to_plink, "Path to Plink")
flags.DEFINE_string('reference_path', temp_path_to_reference, "Path to Reference File")

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
        print("Random sampled traits from dataset, saved as random_vcf_files")
        vcf_files = np.copy(vcf_files_random_choice)
        np.save('random_vcf_files',vcf_files)
        del vcf_files_random_choice
        
    for num_file, a_vcf_file in enumerate(tqdm(vcf_files)):
        with pygwasvcf.GwasVcf(a_vcf_file) as g, pysam.VariantFile(a_vcf_file) as samfile:
            trait_name = g.get_traits()[0]
            trait_rsids_dir = os.path.join(path_to_vcf_files,'{}_rsids'.format(trait_name))
            if not (os.path.isdir(trait_rsids_dir)):
                try:
                    os.mkdir(trait_rsids_dir)
                except OSError:
                    print("Error in making directory")
            else:
                temp_files = os.listdir(trait_rsids_dir)
                if "all_variants" in str(temp_files):
                    print("Skipping trait {}, delete -- todo to make more efficient".format(trait_name))
                    continue # Skipping File
            num_variants = int(g.get_metadata()[trait_name]['TotalVariants']) - int(g.get_metadata()[trait_name]['VariantsNotRead'])
            print("Number of variants {} in the trait {}".format(num_variants, trait_name))
            #print("debug break") -- debugging purposes
            #break
            # gets a list of all the contigs from the header (assumes header info has contigs)
            the_contigs = list(samfile.header.contigs) 
            len_the_contigs = 23 if len(the_contigs) > 23 else len(the_contigs) # Some VCF files contain other information
            rsIDs = np.empty((len_the_contigs,num_variants), dtype=object) # Hopefully no out of index error to be efficient, also not efficient because unknown length of string
            for i, a_contig in enumerate(tqdm(the_contigs)):
                if i > 23:
                    break # VCF files contain other information after 23 chromosomes
                for ind, variant in enumerate(tqdm(g.query(contig=a_contig))):
                    # rsIDs[i, ind]=pygwasvcf.VariantRecordGwasFuns.get_id(variant, trait_name) some id's are '.'
                    temp_trait = pygwasvcf.VariantRecordGwasFuns.get_id(variant, trait_name)
                    if temp_trait != '.':
                        rsIDs[i, ind] = temp_trait # Currently ignore variants without IDs


            # save all rsids into folder of that trait
            rsid_listname='{}/{}_all_variants_file'.format(trait_rsids_dir,trait_name)
            # now choose 
            np.save(rsid_listname, rsIDs)
            print("Finishing saving all rsIDs now moving to smaller subset if flag is not set to 0")
            if num_snps != 0:
                print ("Using a smaller number of varints, {}, in comparison to total number of SNPS, so downsampling SNPs.".format(num_snps))
                newRSIds = np.take(rsIDs, random_unique_indexes_per_row(rsIDs, num_snps))
                rsid_listname='{}/{}_{}_variants_file'.format(trait_rsids_dir,trait_name,num_snps)
                np.save(rsid_listname,newRSIds)
                print("Saved subset of RSIDS of trait {}".format(trait_name))
                del newRSIds
            else:
                raise Exception("Number of variants requested is greater than number of SNPS in dataset, use 0 to request all variants")
            del rsIDs
            print("Now on trait #{}".format(num_file))
            
            

def grabRSID_numpy(path_to_files: str, how_many: int):
    
    import pathlib

    path = pathlib.Path(path_to_files) 
    np_files = path.rglob("*.npy") # get names of all numpy arrays
    pattern_re = re.compile(r"^(?!.*all).*") # Get only files that are sub-samples
    sub_np_files = [a.as_posix() for a in np_files if re.match(pattern_re,a.name)] # Get file name and path
    sub_np_files = sub_np_files[:how_many] # Get a certain number of files

    return sub_np_files
    

def generateLD(path_to_plink: str, path_to_bfile: str, file_list_rsIDs: List[str], ld_threshold: float):
    """Generate an diagonal block matrix consisting of rsIDs, rsIDs on different chromosomes
    are zero, and anything less than threshold will be 0. If shape of rsIDS is too large (X x M),
    M > 1000000 number of SNPS it will require a large memory and hard-drive diskspace
    
    Args:
        rsIDs (list of strings):  list of files that store rsids for traits (saved as numpy ndarray)
        ld_threshold (float): Ld threshold anything below will be zero
    """
    #location of plink /usr/bin/plink1.9
    #location of refernce: /home/ludeep/Desktop/PopGen/eqtlGen/Reference
    def execute_command(command):
        print("Execute command: {}".format(command))
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(process.communicate()[0].decode("utf-8"))
    
    
    # Create a LD matrix for every chromosome
    print("Starting to process each trait rsID to generate LD matrix")
    for an_array in file_list_rsIDs:
        rsIDs = np.load(an_array)
        chromosomes, num_snps = rsIDs.shape
        for contig in range(0,chromosomes):
            snp_list=rsIDs[contig, np.where(rsIDs[contig,:]!=None)]
            # Get Trait Path
            pattern_re = re.compile(r".*\/")
            trait_name_path = re.match(pattern_re,an_array)[0]
            print("Save RSIDS into a text file (one per line for chromosome: {}".format(contig))
            rsid_txt_file_name = '{}contig_{}_rsidFil.txt'.format(trait_name_path,contig)
            
            np.savetxt(rsid_txt_file_name,snp_list,fmt="%s",delimiter='\n') 
            # Create LD file in table format from rsid_ids
            print("Executing Command plink")
            execute_command('{} --bfile {} --r2 triangle gz --extract {} --out {}'.format(path_to_plink, path_to_bfile, rsid_txt_file_name, '{}ld_contig_{}.out'.format(trait_name_path,contig))) 
          
def main(argv):
    
   
    path_to_vcf_files = FLAGS.path_to_vcf_files
    num_traits = FLAGS.num_traits
    num_snps = FLAGS.num_snps
    
    #getRSIDS(path_to_vcf_files, num_traits, num_snps)
    # if passing from bash use: ar1=$(whereis plink | awk '{print $2}')
    # where awk '{print $2}' is the 2nd variable from whereis, which
    # is the path of plink
    file_list = grabRSID_numpy(path_to_vcf_files,1)
    generateLD(FLAGS.plink_path, FLAGS.reference_path,file_list, 1e-8)


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
