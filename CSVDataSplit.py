# -*- coding: utf-8 -*-
# python test.py  C:\Users\Ken\Desktop\Code\TBrain\taetfp.csv
import csv
import sys
import os
import fnmatch


if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding('utf8')
# File names which contains whole data sets
# Obtain files name = ['taetfp','tasharep','tetfp','tsharep']
file_name =[i for i in fnmatch.filter(os.listdir('.'), '*.csv')]
# File names of splited csv
split_file_list = []

def SplitCSV(filename):
    csvreader = csv.reader(open(filename, 'r', newline=''))
    # Header
    colheader = ['Code', 'Date', 'OpenPrice', 'HighestPrice', 'LowestPrice', 'ClosePrice', 'Trading Amount Per Index']
    
    # Recording Stock code
    FirstRow_Flag = 0
    
    # Create a folder for splited data
    # "file_name"Folder >> raw >>
    if not os.path.isdir(filename[:-4] + 'Folder'):
        os.mkdir(filename[:-4] + 'Folder') 
        if 'win' in sys.platform and not os.path.isdir(filename[:-4] + 'Folder\\raw'):
            Path = filename[:-4] + 'Folder\\raw'
        elif 'linux' in sys.platform and not os.path.isdir(filename[:-4] + 'Folder/raw'):
            Path = filename[:-4] + 'Folder/raw'
            
        os.mkdir(Path) 
            
            
    for row_index, row_data in enumerate(csvreader):
        del row_data[2]
        
        # "If" condition for skipping header 
        if FirstRow_Flag == 0:
            FirstRow_Flag = 1
            continue
        
        # Split the whole data into diffetent csv with the stock code
        # If the stock code is different to FirstRow_Flag then create a new writable csv file
        if not row_data[0] == FirstRow_Flag:
            # Different path for different platform
            if 'win' in sys.platform:
                csvwriter = csv.writer(open(filename[:-4] + 'Folder\\raw\\' + row_data[0].strip() +'.csv', 'w', newline=''))
            else:
                csvwriter = csv.writer(open(filename[:-4] + 'Folder/raw/' + row_data[0].strip() +'.csv', 'w', newline=''))
                
            split_file_list.append(row_data[0].strip())
            # Replace the FirstRow_Flag with the new stock code
            FirstRow_Flag = row_data[0]
            
        csvwriter.writerow(row_data)
def main():
    for file in file_name:
        SplitCSV(file)
    
    
if __name__ == '__main__':
    main()