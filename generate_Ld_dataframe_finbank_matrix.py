import glob
from operator import itemgetter
from numpy import random
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
import pickle
import math
import collections
import random
import itertools

FLAGS = flags.FLAGS
'''
Mainly to see if data-frame that contains names of summary statitistics traits
already exists 
'''

#Cluster computer paths
#temp_path_to_files = '/project2/jjberg/data/summary_statistics/Fin_BANK/open_gwas_data_vcf/'
#temp_path_to_reference = '/project2/jjberg/data/1kg/Reference/1kg.v3/EUR/EUR'
#temp_path_to_reference = '/project2/jjberg/data/1kg/plink-files/files/EUR/all_chroms'
#temp_path_to_plink='/software/plink-1.90b6.9-el7-x86_64/plink'

#local computer paths
temp_path_to_reference = '/home/ludeep/Desktop/PopGen/eqtlGen/Reference/1kg.v3/EUR/EUR'
temp_path_to_plink = '/usr/bin/plink1.9'
#temp_path_to_files = '/home/ludeep/Desktop/PopGen/FinBank/open_gwas_data_vcf/'
temp_path_to_files = '/home/ludeep/Desktop/PopGen/FinBank/testing_dirctory/'
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

def generate_dict_summary_stats(path_to_vcf_files: str, num_traits: int,num_snps: int):
    """Get Summary statistics and RSIDs in num_traits vcf files limited to num_snps

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
            len_the_contigs = 23 if len(the_contigs) >= 23 else len(the_contigs) # Some VCF files contain other information
            #rsIDs = np.empty((len_the_contigs,num_variants), dtype=object) # Hopefully no out of index error to be efficient, also not efficient because unknown length of string
            summary_stat_dict = collections.defaultdict(dict)
            for i, a_contig in enumerate(tqdm(the_contigs)):
                summary_stat_dict[i] = collections.defaultdict(dict)
                if i > 23:
                    break # VCF files contain other information after 23 chromosomes
                for ind, variant in enumerate(tqdm(g.query(contig=a_contig))):
                    temp_trait_rsid = pygwasvcf.VariantRecordGwasFuns.get_id(variant, trait_name)
                    summary_stat_dict[i][variant.pos]['beta'] = pygwasvcf.VariantRecordGwasFuns.get_beta(variant, trait_name)
                    summary_stat_dict[i][variant.pos]['se'] = pygwasvcf.VariantRecordGwasFuns.get_se(variant, trait_name)
                    summary_stat_dict[i][variant.pos]['af'] = pygwasvcf.VariantRecordGwasFuns.get_beta(variant, trait_name)
                    summary_stat_dict[i][variant.pos]['rsid'] = temp_trait_rsid # Note some RSIDS are '.'
                    # rsIDs[i, ind]=pygwasvcf.VariantRecordGwasFuns.get_id(variant, trait_name) some id's are '.'
                    #if temp_trait != '.':
                    #    rsIDs[i, ind] = temp_trait # Currently ignore variants without IDs


            # save all summmary stats dictionary into folder of that trait
            summary_state_dict_name='{}/{}_all_variants_file'.format(trait_rsids_dir,trait_name)
             
            np.save(summary_state_dict_name, summary_stat_dict)
            print("Finishing saving all rsIDs now moving to smaller subset if flag is not set to 0")
            if num_snps != 0:
                print ("Using a smaller number of varints, {}, in comparison to total number of SNPS, so downsampling SNPs.".format(num_snps))
                summary_state_dict_name = '{}/{}_{}_variants_file'.format(trait_rsids_dir,trait_name,num_snps)
                summary_state_dict_name_smaller = collections.defaultdict(dict)
                # Extract SNPS from dictionary
                for contig in range(0, 23):
                    sub_keys = random.sample(list(summary_stat_dict[contig].keys()), int(num_snps / 23)) # random subsample of keys
                    sub_dict = itemgetter(*sub_keys)(summary_stat_dict[contig]) # list of dictionary of beta/allele freq/rsid/se values
                    summary_state_dict_name_smaller[contig]=dict.fromkeys(sub_keys)
                    # could use cython for double speedboost see: 
                    # https://stackoverflow.com/questions/45882166/performance-of-updating-multiple-key-value-pairs-in-a-dict
                    for key, val in zip(sub_keys, sub_dict):  
                        summary_state_dict_name_smaller[contig][key] = val

                np.save(summary_state_dict_name, summary_state_dict_name_smaller)
                print("Saved subset of summary stats of trait {}".format(trait_name))
                del summary_state_dict_name_smaller
                del summary_state_dict_name, sub_keys, sub_dict
            else:
                raise Exception("Number of variants requested is greater than number of SNPS in dataset, use 0 to request all variants")
            del summary_stat_dict
            print("Now on trait #{}".format(num_file))
            
            

def grabRSID_numpy(path_to_files: str, how_many: int):
    
    import pathlib

    path = pathlib.Path(path_to_files) 
    np_files = path.rglob("*.npy") # get names of all numpy arrays
    pattern_re = re.compile(r"^(?!.*all).*") # Get only files that are sub-samples
    sub_np_files = [a.as_posix() for a in np_files if re.match(pattern_re,a.name)] # Get file name and path
    sub_np_files = sub_np_files[:how_many] # Get a certain number of files

    return sub_np_files
    

def grab_all_variant_path(path_to_files: str):
    
    import pathlib
    # testing path
    #test_path = '/home/ludeep/Desktop/PopGen/FinBank/testing_dirctory/'
    #path = pathlib.Path(test_path)  
    path = pathlib.Path(path_to_files) 
 
    save_path = "{}/common_snps/".format(path)
    if not (os.path.isdir(save_path)):
        try:
            os.mkdir(save_path)
        except OSError:
            print("Error in making common snp  directory")
    
    
    np_files = path.rglob("*.npy") # get names of all numpy arrays
    pattern_re = re.compile(r".*?(all_variants).*?") # Get only files that have all SNPS 
    np_dicts = [a.as_posix() for a in np_files if re.match(pattern_re, a.name)] # Get file name and path
    file_save_path = "{}/path_to_variants".format(save_path) # note this should be eventually changed because of the
    # regex above that looks for "all_variants"
    np.save(file_save_path, np_dicts)
    
    return np_dicts, file_save_path

def grabcommon_SNPS(path_to_files: str=None, list_of_paths: str=None, how_many: int=None):
    """Grab common SNPS between traits set by chromosome:position

    Args:
        path_to_files (str, optional): _description_. Defaults to None.
        list_of_paths (str, optional): _description_. Defaults to None.
        how_many (int, optional): _description_. Defaults to None.
    """    
    if list_of_paths is None:
        np_dicts, list_of_paths = grab_all_variant_path(path_to_files)
    else:
        np_dicts = np.load(list_of_paths, allow_pickle=True)
    
    print("Have found {} traits to find common SNPS".format(len(np_dicts)))

    common_snps_dict = collections.defaultdict(dict)
    import pdb
    #Load files
    for i in tqdm(range(0, len(np_dicts)-1)):
        if i == 0:
            dict1 = np.load(np_dicts[i], allow_pickle=True).item()
            dict2 = np.load(np_dicts[i+1], allow_pickle=True).item()
        else:
            dict2 = np.load(np_dicts[i+1], allow_pickle=True).item()
        for contig in range(0,23):  # should be upto 23 chromosomes
            if not common_snps_dict[contig]: 
                print("contig {}".format(contig))
                #pdb.set_trace()
                common_snps_dict[contig] = dict1[contig].keys() & dict2[contig].keys()
            else: # common snps have already been found between first two dictionaries, so now only update with an interesection of the newest dictionary
                check_common_empty = common_snps_dict[contig].intersection(dict2[contig].keys())
                if check_common_empty:
                    common_snps_dict[contig] = check_common_empty
                else:
                    print("No intersections found in this trait {}:".format(np_dicts[i+1]))
        if i == 0:
            del dict1, dict2
        else:
            del dict2
                
    print("finished finding common SNPS")
    #save_path = "{}/common_snps/".format(path_to_files)
    np.save('common_snps_dict', common_snps_dict)
        

def convert_dict_chr_to_rsid(path_to_files: str=None, list_of_paths: str=None, how_many: int=None):
    """Utility function to convert dictionary that is keyed by chromome:position to rsid

    Args:
        path_to_files (str, optional): _description_. Defaults to None.
        list_of_paths (str, optional): _description_. Defaults to None.
        how_many (int, optional): _description_. Defaults to None.
    """
    if list_of_paths is None:
        np_dicts, list_of_paths = grab_all_variant_path(path_to_files)
    else:
        np_dicts = np.load(list_of_paths, allow_pickle=True)
    
    
    import pathlib

    save_path = "{}rsid_summary_stat_dicts/".format(path_to_files)
    if not (os.path.isdir(save_path)):
        try:
            os.mkdir(save_path)
        except OSError:
            print("Error in creating dictory for converting summary stat dict to RSIDS")
    
    rsid_stat_dict = collections.defaultdict(dict)
    for i in tqdm(range(0, len(np_dicts)-1)):
        skipped = 0   # counter for how many variants were skipped
        trait_dict = np.load(np_dicts[i], allow_pickle=True).item()  # load trait dictionary
        for contig in trait_dict.keys():
            for position in trait_dict[contig].keys():
                temp_rsid = trait_dict[contig][position]['rsid']
                if temp_rsid != '.':
                    rsid_stat_dict[temp_rsid]['beta'] =  trait_dict[contig][position]['beta']
                    rsid_stat_dict[temp_rsid]['se'] =  trait_dict[contig][position]['se']
                    rsid_stat_dict[temp_rsid]['af'] =  trait_dict[contig][position]['af']
                else:
                    skipped += 1
        rsid_stat_dict['skipped'] = skipped
        # Saving RSID trait 
        # RSID file name
        # Grab Trait name
        trait_name = np_dicts[i].split('/')[-1] # trait name is last part of file path
        rsid_file_name = "{}{}_rsids".format(save_path,trait_name[:-4])
        np.save(rsid_file_name, rsid_stat_dict)
            
        
        
        

def grabcommon_SNPS_RSID(path_to_files: str=None, list_of_paths: str=None, how_many: int=None):
    """Grab common SNPS between traits sorted by RSIDS

    Args:
        path_to_files (str, optional): _description_. Defaults to None.
        list_of_paths (str, optional): _description_. Defaults to None.
        how_many (int, optional): _description_. Defaults to None.
    """ 

    if list_of_paths is None:
        np_dicts, list_of_paths = grab_all_variant_path(path_to_files)
    else:
        np_dicts = np.load(list_of_paths, allow_pickle=True)
    
    print("Have found {} traits to find common SNPS via RSIDS".format(len(np_dicts)))

    common_snps_dict = collections.defaultdict(dict)
    import pdb
    #Load files
    for i in tqdm(range(0, len(np_dicts)-1)):
        if i == 0:
            dict1 = np.load(np_dicts[i], allow_pickle=True).item()
            dict2 = np.load(np_dicts[i+1], allow_pickle=True).item()
        else:
            dict2 = np.load(np_dicts[i+1], allow_pickle=True).item()
        for contig in range(0,23):  # should be upto 23 chromosomes
            if not common_snps_dict[contig]: 
                print("contig {}".format(contig))
                #pdb.set_trace()
                
            else: # common snps have already been found between first two dictionaries, so now only update with an interesection of the newest dictionary
                check_common_empty = common_snps_dict[contig].intersection(dict2[contig].keys())
                if check_common_empty:
                    common_snps_dict[contig] = check_common_empty
                else:
                    print("No intersections found in this trait {}:".format(np_dicts[i+1]))
        if i == 0:
            del dict1, dict2
        else:
            del dict2
            
def generateLD(path_to_plink: str, path_to_bfile: str, file_list_rsIDs: List[str], ld_threshold: float=1e-8):
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


def generateLD_SummaryStats(path_to_plink: str, path_to_bfile: str, common_snps_path: str, ld_threshold: float=1e-8):
    """Generate an diagonal block matrix LD matrix of snps identified by chromosome and variant position
    anything outside that chromosome regions is 0, and anything less than threshold will be 0. If shape of matrix is too large (X x M),
    M > 1000000 number of SNPS it will require a large memory and hard-drive diskspace
    
    Args:
        common_snps (dictionary of sets):  a path to a dictionary of sets where each set is a variant position and each key is the chromosome
        ld_threshold (float): Ld threshold anything below will be zero
    """
    #location of plink /usr/bin/plink1.9
    #location of refernce: /home/ludeep/Desktop/PopGen/eqtlGen/Reference
    def execute_command(command):
        print("Execute command: {}".format(command))
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(process.communicate()[0].decode("utf-8"))
    
    
    # Create a LD matrix for every chromosome
    print("Starting to process a list of common variants to generate LD matrix")
    
    common_snps = np.load(common_snps_path, allow_pickle=True).item()
    assert len(common_snps.keys()) == 23, "Number of chromosomes in dictionary should only be 23"
    
    # Create SNP text file where column 1 is chromosome,
    # Columns 2 and 3 are [f1st base pair position, 2nd base pair position (can be the same)]
    # Column 4 is an arbritatry label
    ld_save_path = 'Ld_calculations3'
    if not (os.path.isdir(ld_save_path)):
        try:
            os.mkdir(ld_save_path)
        except OSError:
            print("Error in making directory")
    
    #lambda function generate column 4
    col4_fun = lambda x: str(x[0][0])+':'+str(x[1][0])
    with open('{}/snp_list_for_ld.txt'.format(ld_save_path), 'ab') as the_file:
        for contig in tqdm(common_snps.keys()):
            snp_list = np.expand_dims(np.asarray(list(common_snps[contig]), dtype='O'), axis=1)
            if contig < 22:
                actual_contig = contig + 1 # Since we are starting from index 0 
                contig_col = np.expand_dims(np.asarray([actual_contig]*len(snp_list),dtype='O'),axis=1)  # this is the first column 
                col4_names = np.expand_dims(np.asarray(list(map(col4_fun,zip(contig_col,snp_list))),dtype='O'),axis=1)
                thearray = np.concatenate((contig_col,snp_list, snp_list, col4_names),axis=1)
                np.savetxt(the_file, thearray, fmt='%s')
            else:
                actual_contig='X'
                contig_col = np.expand_dims(np.asarray(['X']*len(snp_list),dtype='O'),axis=1) 
                col4_names = np.expand_dims(np.asarray(list(map(col4_fun,zip(contig_col,snp_list))),dtype='O'),axis=1)
                thearray = np.concatenate((contig_col,snp_list, snp_list, contig_col),axis=1)
                np.savetxt(the_file, thearray, fmt='%s')
            print("Executing Command plink to get LD matrix for contig: {}".format(actual_contig))
            execute_command('{} --bfile {} --r2 triangle gz --extract range {} --out {}'.format
                    (path_to_plink, path_to_bfile, '{}/snp_list_for_ld.txt'.format(ld_save_path),
                    '{}/ld_contig_{}.out'.format(ld_save_path,actual_contig))) 
    
    

def generate_common_reference_snps(path_to_plink: str, path_to_bfile: str, common_snps_path: str):
    """[summary]

    Args:
        path_to_plink (str): path to executable plink path
        path_to_bfile (str): path to reference file
        common_snps_path (str): a path to a dictionary of sets where each set is a variant position and each key is the chromosome
    """
    
    #location of plink /usr/bin/plink1.9
    #location of refernce: /home/ludeep/Desktop/PopGen/eqtlGen/Reference
    def execute_command(command):
        print("Execute command: {}".format(command))
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(process.communicate()[0].decode("utf-8"))
    
    # Create a common list for every chromosome
    
    # Create a LD matrix for every chromosome
    print("Starting to process a list of common variants to common variants from Reference")
    
    common_snps = np.load(common_snps_path, allow_pickle=True).item()
    assert len(common_snps.keys()) == 23, "Number of chromosomes in dictionary should only be 23"
    
    # Create SNP text file where column 1 is chromosome,
    # Columns 2 and 3 are [f1st base pair position, 2nd base pair position (can be the same)]
    # Column 4 is an arbritatry label
    snp_reference_common = 'Reference_Common_Snps'
    if not (os.path.isdir(snp_reference_common)):
        try:
            os.mkdir(snp_reference_common)
        except OSError:
            print("Error in making directory")
    
    #lambda function generate column 4
    col4_fun = lambda x: str(x[0][0])+':'+str(x[1][0])
    with open('{}/snp_list_for_reference.txt'.format(snp_reference_common), 'ab') as the_file:
        for contig in common_snps.keys():
            snp_list = np.expand_dims(np.asarray(list(common_snps[contig]), dtype='O'), axis=1)
            if contig < 22:
                actual_contig = contig + 1 # Since we are starting from index 0 
                contig_col = np.expand_dims(np.asarray([actual_contig]*len(snp_list),dtype='O'),axis=1)  # this is the first column 
                col4_names = np.expand_dims(np.asarray(list(map(col4_fun,zip(contig_col,snp_list))),dtype='O'),axis=1)
                thearray = np.concatenate((contig_col,snp_list, snp_list, col4_names),axis=1)
                np.savetxt(the_file, thearray, fmt='%s')
            else:
                actual_contig='X'
                contig_col = np.expand_dims(np.asarray(['X']*len(snp_list),dtype='O'),axis=1) 
                col4_names = np.expand_dims(np.asarray(list(map(col4_fun,zip(contig_col,snp_list))),dtype='O'),axis=1)
                thearray = np.concatenate((contig_col,snp_list, snp_list, contig_col),axis=1)
                np.savetxt(the_file, thearray, fmt='%s')
            print("Executing Command plink to get common snps for contig: {}".format(contig))
            execute_command('{} --bfile {} --extract range {} --write-snplist'.format
                    (path_to_plink, path_to_bfile, '{}/snp_list_for_reference.txt'.format(snp_reference_common))) 
            os.rename('plink.snplist','{}/common_snp_contig_{}.snplist'.format(snp_reference_common, actual_contig) )
            execute_command('{} --bfile {} --extract range {} --write-snplist {}'.format
                    (path_to_plink, path_to_bfile, 'snp_list_for_reference.txt', '{}/common_snp_contig_{}.out'.format(snp_reference_common,contig))) 


def generateSummaryStats(unique_snps_path: str, path_to_vcf_files: str, random_sub_sample: None):
    """
        This function unfortunately does not work, as it is difficult to query a VCF file based on RSID.  
        
    Args:
        unique_snps (str): Path to unique set of SNPS identified by RSIDS
        path_to_vcf_files (str): [description]
        random_sub_sample (None): [description]

    Raises:
        Exception: [description]
    """    
    
    summary_dict = {}
    if not random_sub_sample:
        random_sub_sample = math.inf
    
    vcf_files = glob.glob("{}*.gz".format(path_to_vcf_files))
    with open(unique_snps_path) as file:
        for line in file:
            regex_pattern = r"[a-z0-9]*[^_;]"
            rsid = line.rstrip().strip()
            matches = re.finditer(regex_pattern, rsid)
            for matchNum, match in enumerate(matches):
                print ("Match {matchNum} was found at: {match}".format(matchNum = matchNum, match = match.group()))
                summary_dict[match.group()] = None
            if len(summary_dict) > random_sub_sample:
                break
    
    for num_file, a_vcf_file in enumerate(tqdm(vcf_files)):
        with pygwasvcf.GwasVcf(a_vcf_file) as g, pysam.VariantFile(a_vcf_file) as samfile:
            trait_name = g.get_traits()[0]
            trait_rsids_dir = os.path.join(path_to_vcf_files,'{}_rsids'.format(trait_name))
            g.index_rsid()
            if not (os.path.isdir(trait_rsids_dir)):
                try:
                    os.mkdir(trait_rsids_dir)
                except OSError:
                    print("Error in making directory")
            print("Getting statistics for this number of variants {} in the trait {}".format(len(summary_dict), trait_name))
            #print("debug break") -- debugging purposes
            #break
            # gets a list of all the contigs from the header (assumes header info has contigs)
            for i, the_rsid in enumerate(tqdm(summary_dict.keys())):
                variant = g.query(variant_id=the_rsid)
                contig, pos = g.get_location_from_rsid(rsid=the_rsid)
                # print variant-trait SE
                summary_dict[the_rsid]['se'] = pygwasvcf.VariantRecordGwasFuns.get_se(variant, trait_name)
                # print variant-trait beta
                summary_dict[the_rsid]['beta'] = pygwasvcf.VariantRecordGwasFuns.get_beta(variant, trait_name)
                # print variant-trait allele frequency
                summary_dict[the_rsid]['af'] = pygwasvcf.VariantRecordGwasFuns.get_af(variant, trait_name)
            
            # save all rsids into folder of that trait
            summary_dict_name='{}/{}_summary_stats.pkl'.format(trait_rsids_dir,trait_name)
             
            with open(summary_dict_name, 'wb') as f:
                pickle.dump(summary_dict, f)
            
            print("Finished saving summary stats to dictionary")
            
            
    
        
          
def main(argv):
    
   
    path_to_vcf_files = FLAGS.path_to_vcf_files
    num_traits = FLAGS.num_traits
    #num_traits = 1 # For testing
    num_snps = FLAGS.num_snps
    
    #testing calls
    #generate_dict_summary_stats(path_to_vcf_files, num_traits, num_snps)
    #grabcommon_SNPS('na', 'path_of_all_variants.npy')
    #common_snps_path = 'common_snps.npy'
    #generateLD_SummaryStats(FLAGS.plink_path, FLAGS.reference_path,'common_snps.npy', 1e-8)
    
    #Cluster calls
    #generate_dict_summary_stats(path_to_vcf_files, num_traits, num_snps)
    #grabcommon_SNPS(path_to_vcf_files, '/project2/jjberg/data/summary_statistics/Fin_BANK/open_gwas_data_vcf/common_snps/path_to_variants.npy')
    #grabcommon_SNPS(path_to_vcf_files)
    #common_snps_path = 'common_snps.npy'
    #generateLD_SummaryStats(FLAGS.plink_path, FLAGS.reference_path, common_snps_path, 1e-8)
    convert_dict_chr_to_rsid(path_to_vcf_files)
    #generate_common_reference_snps(FLAGS.plink_path, FLAGS.reference_path, common_snps_path)

    #common_snps_path = 'common_snps_dict.npy'
    #generateLD_SummaryStats(FLAGS.plink_path, FLAGS.reference_path, common_snps_path, 1e-8)

    # if passing from bash use: ar1=$(whereis plink | awk '{print $2}')
    # where awk '{print $2}' is the 2nd variable from whereis, which
    # is the path of plink
    #file_list = grabRSID_numpy(path_to_vcf_files,100)
    #generateLD(FLAGS.plink_path, FLAGS.reference_path,file_list, 1e-8)
    #generateSummaryStats('/home/ludeep/Desktop/PopGen/FinBank/rsid_temp/contig_0_rsidFil.txt', path_to_vcf_files, 1000)



if __name__ == '__main__':
    app.run(main)
    
