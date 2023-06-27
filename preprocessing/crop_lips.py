import os
from multiprocessing import Pool
from preprocess_script import preprocess_data

SPLITS = ['pretrain', 'test', 'train', 'val']


if __name__ =='__main__':
    
    EXTENSION = '.mp4'

    for split in SPLITS:
        # Read the text file while removing \n
        relative_dataset_folder = os.path.join('.', '..', 'data', 'lrs2_v1', 'mvlrs_v1')
        absolute_path = os.path.abspath(relative_dataset_folder)
        
        dataset_file = os.path.join('.', '..', 'data', 'lrs2_v1', 'mvlrs_v1', split)
        with open(f'{dataset_file}.txt', 'r') as file:
            lines = [line.rstrip() for line in file.readlines()]
            
        if split == 'pretrain':
            folder = 'pretrain'
        else:
            folder = 'main'


        pool = Pool()
            
        pool.starmap(preprocess_data, [(line, folder, absolute_path) for line in lines])
        
        # Close the pool to free resources
        pool.close()
        pool.join()