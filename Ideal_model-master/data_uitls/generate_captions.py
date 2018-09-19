# -*- coding: utf-8 -*-


import os

# captions = list()
data_directory = '/Users/apple/Desktop/Ideal_model/raw_data/img'
# data_directory ='/path/to/your/data/'
count = 0

for fn in os.listdir(data_directory):
    count = count +1

Num_of_data = count

# print count
# For every image in the data set, generate captions using NIC model.
for i in range(Num_of_data - 1):
    image_file_name = '000' + format(str(i), '0>3s') + '.jpg'
    # nic_cmd = 'python ~/run_inference.py --input_files /path/to/your/data/%%%.jpg'
    nic_cmd = 'python /Users/apple/Desktop/IndividualProject/image_caption/runned_model/Show_And_Tell/run_inference.py --input_files /Users/apple/Desktop/Ideal_model/raw_data/img/' + image_file_name
    os.system(nic_cmd + ' | ' +'tee -a '+' output.txt')
    print 'caption for' + image_file_name + 'added into output file'
    
print '=================================='    
print 'all images are dealed with caption added by NIC'
print '==================================' 

