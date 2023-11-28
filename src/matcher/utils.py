"Utils method for to help utilizing package functionalities "
import os
from glob import glob
import pandas as pd


def read_files(
    path:str, 
    files_pattern:str,
    col_names:list,
    sep:str='\t'
):
    """read and concat list of files

    @params
    -------
    path(str): directory of readable files or path of one file
    files_pattern(str): str expected to be in the files required to read
    col_name(List[str]): columns names in the files
    
    @return
    -------
    pd.DataFrame for all files together
    """
    if os.path.isdir(path):
        files =  glob(f'{path}/*')
    elif o.path.isfile(path):
        return pd.read_csv(path)
    else: 
        raise FileExistsError(f"This path `{path}` not exist")
 
    
    filtered_files = []
    for each_file in files:
        if each_file.__contains__(files_pattern):  #since its all type str you can simply use startswith
            filtered_files.append(each_file)

    print('>> Num of files:',len(filtered_files))
    all_filtered = pd.concat([pd.read_csv(f'{f}',low_memory=False,sep=sep, names=col_names) for f in filtered_files],axis=0)
    all_filtered.drop(all_filtered[all_filtered[col_names[0]] == col_names[0]].index,inplace=True)
    all_filtered= all_filtered.dropna().reset_index(drop=True)    
    return all_filtered