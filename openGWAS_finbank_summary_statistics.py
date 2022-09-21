import ieugwaspy as igd
import numpy as np
import pandas as pd
import os
from typing import TypeVar
import urllib
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
flags.DEFINE_string('dataframe_name', 'fin_biobank_summary_state.pkl', 'Dataframe (pkl) File Name')
flags.DEFINE_string('path_to_store_files', 'open_gwas_data_vcf/', 'Path to store summary statistics')
flags.DEFINE_integer('num_traits', 10, 'Number of traits to download')
flags.DEFINE_string('gwas_id', 'finn-b', 'Dataset ID corresponding to openGWAS')


PandasDataFrame = TypeVar('pandas.core.frame.DataFrame')

def download_progress(block_num, block_size, total_size):
    """Visualize download progress of file
    Reference: http://rmehta1987.github.io/tips/2021-10-26-urllib_urlretrieve_downloadhook/
    Args:
        block_num ([int]): [description]
        block_size ([int]): [description]
        total_size ([int]): [description]
    """
    per=100.0*block_num*block_size/total_size 
    if per>100:  
        per=100  
    print('{:.2f}'.format(per))     

def download_url(url: str, root:str, filename: str=None):
    """Download a file from a url and place it in root.
    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the basename of the URL
    """

    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    os.makedirs(root, exist_ok=True)

    try:
        print('Downloading ' + url + ' to ' + fpath)
        urllib.request.urlretrieve(url, fpath, reporthook=download_progress)
    except (urllib.error.URLError, IOError) as e:
        if url[:5] == 'https':
            url = url.replace('https:', 'http:')
            print('Failed download. Trying https -> http instead.'
                    ' Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(url, fpath, reporthook=download_progress)
            
def get_data(get_summary_stat_name: str='summary_stat_name.pkl', gwas_id: str='finn-b', population: str='European'):
    """Get a list of GWAS samples from OpenGWAS API.
    Args:
        get_summary_stat_name = Save dataframe of GWAS's as returned by API
        gwas_id = Which GWAS id to find
        population = Which population type to downloads
    """
    try:
        if os.path.isfile(get_summary_stat_name):
            summary_df =  pd.read_pickle(get_summary_stat_name)
        else:
            data = igd.gwasinfo() # This gets all studies in the database
            gwas_df = pd.DataFrame.from_dict(data, orient='index')  # reorientating data to one study per row

            # Get all the eqtl-a studies (https://gwas.mrcieu.ac.uk/datasets/?gwas_id__icontains=eqtl-a)
            summary_df = gwas_df[gwas_df["id"].str.contains(gwas_id, case=False) 
                        & gwas_df["population"].str.contains(population, case=False)]
           
            # Pickle/save the dataframe
            summary_df.to_pickle(get_summary_stat_name)
    except (IOError, OSError):
        print("Writing to file failed, check permissions")

    return summary_df

# Download GWAS-VCF files 
def download_gwas_vcf(summary_stats_df: PandasDataFrame, num_trait_download: int=100, gwas_id: str='finn-b', file_root: str='data/'):
    """Download summary statistics VCF from OpenGWAS.
    Args:
        dataframe = dataframe containing openGWAS dataframe
        num_trait_download = number of datasets to download
        file_root = where to download
    """
    # format wget https://gwas.mrcieu.ac.uk/files/ukb-b-19953/ukb-b-19953.vcf.gz (gwas-vcf files)
    # format wget https://gwas.mrcieu.ac.uk/files/ukb-b-19953/ukb-b-19953.vcf.gz.tbi (index files)
    if num_trait_download > 0:
        sampled_summary_df = summary_stats_df['id'].sample(n=num_trait_download,random_state=1).tolist()
    else:
        sampled_summary_df = summary_stats_df['id'].tolist()  # Get all available traits
    try:
        if not os.path.exists(file_root):
            os.makedirs(file_root)
    except (IOError, OSError):
        print("Writing to file failed, check permissions")
        
    for id in tqdm(sampled_summary_df):
        
        id_file_name = '{}{}.vcf.gz'.format(file_root,id)
        if not os.path.exists(id_file_name):
            # download vcf
            #dl_url = 'https://gwas.mrcieu.ac.uk/files/{}/{}.vcf.gz'.format(id,id)
            # Using oracle: https://objectstorage.us-ashburn-1.oraclecloud.com/n/idrvm4tkz2a8/b/opengwas/o/<batch>/<dataset>/<filename.{vcf.gz,vcf.gz.tbi,_report.html}>
            dl_url = 'https://objectstorage.us-ashburn-1.oraclecloud.com/n/idrvm4tkz2a8/b/opengwas/o/{}/{}/{}.vcf.gz'.format(gwas_id, id, id)
            print("Downloading vcf file: {}".format(id))
            download_url(dl_url, file_root)
            
            # download index
            #dl_url = 'https://gwas.mrcieu.ac.uk/files/{}/{}.vcf.gz.tbi'.format(id,id)
            dl_url = 'https://objectstorage.us-ashburn-1.oraclecloud.com/n/idrvm4tkz2a8/b/opengwas/o/{}/{}/{}.vcf.gz.tbi'.format(gwas_id, id, id)
            print("Downloading vcf file: {}".format(id))
            download_url(dl_url, file_root)
        else:
            print("Skipping file as it already exists: {}".format(id))
   
def main(argv):
    
    df_name = FLAGS.dataframe_name
    path_to_store_files = FLAGS.path_to_store_files
    num_traits = FLAGS.num_traits
    gwas_id = FLAGS.gwas_id
    
    if df_name is None:
        print("No filename supplied, assuming file does not exist, getting summary statistics names and downloading")
        summary_stats_df = get_data('fin_biobank_summary_state.pkl')
        download_gwas_vcf(summary_stats_df, num_traits, gwas_id, path_to_store_files)
    else:
        try:
            if not os.path.exists(df_name):
                print("File does not exist, resampling dataset, default save file {}".format('fin_biobank_summary_state.pkl'))
                summary_stats_df = get_data(
                    'fin_biobank_summary_state.pkl')
                download_gwas_vcf(summary_stats_df, num_traits, gwas_id, path_to_store_files)
            else:
                summary_stats_df = pd.read_pickle(df_name)
                download_gwas_vcf(summary_stats_df, num_traits, gwas_id, path_to_store_files)
                print("File exists, restarting download")
        except (IOError, OSError):
            print("Writing to file failed, check permissions")
            
if __name__ == '__main__':
    app.run(main)




