# -*- coding: utf-8 -*-
import csv
import sys
import os
import numpy as np
from pandas.core.frame import DataFrame
import fnmatch

if sys.version[0] == '2':
    sys.reload(sys)
    sys.setdefaultencoding('utf8')
# File names which contains whole data sets
# Obtain files name = ['taetfp','tasharep','tetfp','tsharep']
file_name =[i[:-4] for i in fnmatch.filter(os.listdir('.'), '*.csv')]
# For RSI(): Relative Strength Index
rsi_list = [3,7,12,26,60]
# For MA(): Moving average list
ma_list = [3,7,12,26,60]
# For BIAS(): Moving average list
bias_list = [3,7,12,26,60]
# For WMA(): Weighted moving average list
wma_list = [3,7,12,26,60]
# For ES(): Exponential smoothing rate
es_forecast_rate = np.arange(0,1,0.05)
# For %K, %D, slow %D, [2,4,6,8,10]
stochastic_interval = range(4, 12, 2)
#For Momentum
momentum_list=range(5, 20, 5)
#For MACD
macd_list=range(5, 20, 5)
# For MA_Vol(): Moving average (volume) list
ma_vol_list = [3,7,12,26,60]
#For Ratio_Vol:
ratio_vol_list = [3,7,12,26,60]


def feature(folder):
    
    if 'win' in sys.platform:
        Path = folder + 'Folder\\raw\\'
    else:
        Path = folder + 'Folder/raw/'
    new_Path = Path.replace('raw','featured')
    if not os.path.isdir(new_Path):
        os.mkdir(new_Path)
    
    # Walk all over the folder    
    Split_files = os.listdir(Path)

    # Loop of all files
    for split in Split_files:
        
        # Header
        colheader = ['Code', 'Date', 'OpenPrice', 'HighestPrice', 'LowestPrice', 'ClosePrice', 'Volume']
        
        # Read Raw data
        # 0.'Code'
        # 1.'Date'
        # 2.'OpenPrice' 
        # 3.'HighestPrice' 
        # 4.'LowestPrice'
        # 5.'ClosePrice'
        # 6.'Trading Amount Per Index'
        csvreader = csv.reader(open(Path + split,'r'))

        # Initialize the list
        new_rsi_list = []
        new_ma_list = []
        new_wma_list = []
        high_list = []
        low_list = []
        D_list = []
        slow_D_list = []
        new_momentum_list=[]
        new_macd_list=[]
        new_bias_list=[]
        k=1
        new_ma_vol_list=[]
        new_ratio_vol_list=[]
        # Change the raw data type to list
        raw_list = list(csvreader)
        
        
#Initialize
        # Remove ',' and blank of the raw data
        for i in range(len(raw_list)):
            #print (len(raw_list))
            #print (len(raw_list(0)))
            for j in range(len(raw_list[i])):
                raw_list[i][j] = raw_list[i][j].strip()
                raw_list[i][j] = raw_list[i][j].replace(',','')
                
# High-Low
# Close-Open
        colheader.append('High-Low')
        colheader.append('Close-Open')
        for index, data in enumerate(raw_list):
            raw_list[index].append(float(data[3])-float(data[4]))     
            raw_list[index].append(float(data[5])-float(data[2])) 
# RSI: Relative Strength Index
        # Pick the rsi list
        for content in rsi_list:
            # The rsi can't work if the number of records is lower than the rsi numbers.
            # Pick the rsi which is lower than the number of records
            if len(raw_list) > content:
                # new_rsi_list is a list keeps proper number of rsi
                new_rsi_list.append(content)
        # Moving average      
        for rsi in new_rsi_list:
            # "elements" list contains the differ between close prices of data 
            elements = []
            # Append a new header
            colheader.append('close-rsi %' + str(rsi))
            ex_value = 0
            '''
            for index, data in enumerate(raw_list):
                if index == 0:
                    ex_value = float(data[5])
                    raw_list[index].append('')
                else:
                    elements.append(float(data[5])-ex_value)
                    ex_value = float(data[5])
                    # element is full
                    
                    if len(elements) == rsi:
                        check_list = [i for i in elements if not i == 0]
                        if check_list:
                            raise_value = sum([round(i, 4) for i in elements if i > 0]) / rsi
                            fall_value = sum([round(i, 4) for i in elements if i < 0]) / rsi * -1
                            rsi_value = raise_value / (raise_value + fall_value) * 100
                        del elements[0]
                        raw_list[index].append(rsi_value)
                    else:
                        raw_list[index].append('')
                        print(split, index, data)
                        print(elements)
            '''
            for index, data in enumerate(raw_list):
                if index <= rsi:
                    if index > 0:
                        elements.append(float(data[5])-ex_value)
                    ex_value = float(data[5])
                    raw_list[index].append('')
                else:
                    raise_value = sum([round(i, 4) for i in elements if i > 0]) / rsi
                    fall_value = sum([round(i, 4) for i in elements if i < 0]) / rsi * -1
                    if raise_value + fall_value > 0:
                        rsi_value = raise_value / (raise_value + fall_value) * 100
                    elements.append(float(data[5])-ex_value)
                    del elements[0]
                    raw_list[index].append(rsi_value)
                    ex_value = float(data[5])
                    '''
                    print(elements)
                    print(rsi_value)
                os.system('pause')
                    '''
# Moving average                
        # Pick the ma list
        ma_number =[]
        for content in ma_list:
            # The MA can't work if the number of records is lower than the ma numbers.
            # Pick the ma which is lower than the number of records
            if len(raw_list) > content:
                # new_ma_list is a list keeps proper number of ma
                new_ma_list.append(content)
                
        # Moving average      
        for ma in new_ma_list:
            # "elements" list contains ma number of data 
            elements = []
            # Append a new header
            colheader.append('close-ma' + str(ma))
            #record the position of ma for bias caculation
            ma_number.append(len(colheader))
            
            for index, data in enumerate(raw_list):
                elements.append(float(data[5]))
                #ma_elements[0, index].append(float(data[5]))
                # If the "elements" list contains ma number of close value, then calculate the ma value
                if len(elements) == ma:
                    # Append the ma value to the botton of data list
                    raw_list[index].append(np.mean(elements))
                    #ma_elements[1, index].append(np.mean(elements))

                    del elements[0]
                else:
                    raw_list[index].append('')
# BIAS               
        # Pick the BIAS list
        
        for content in bias_list:
            # The BIAS can't work if the number of records is lower than the bias numbers.
            # Pick the bias which is lower than the number of records
            if len(raw_list) > content:
                # new_bias_list is a list keeps proper number of ma
                new_bias_list.append(content)
               
        # BIAS     
        for bias in new_bias_list:

            # "elements" list contains bias number of data 
            elements = []
                # Append a new header
            colheader.append('close-bias' + str(bias))
            #Get the position of ma value in raw list 
            j=ma_number[k-1]
            #position move
            k=k+1
                #print (len(colheader))
            for index, data in enumerate(raw_list):

                elements.append(float(data[5]))
                
                    # If the "elements" list contains ma number of close value, then calculate the ma value
                if len(elements) == bias:

                    #save bias in raw_list
                    raw_list[index].append((float(data[5])-float(data[j-1]))/float(data[j-1]))
                        # Delete the first close value of element
                    del elements[0]
                    
                else:
                    raw_list[index].append('')     
                                   
# Weighted moving average        
        # Pick the wma list
        for content in wma_list:
            # The WMA can't work if the number of records is lower than the wma numbers.
            # Pick the wma which is lower than the number of records
            if len(raw_list) > content:
                new_wma_list.append(content) 
                
        # Weighted moving average                   
        for wma in new_wma_list:
            # "elements" list contains wma number of data 
            elements = []
            # Append a new header
            colheader.append('close-wma' + str(wma))  
            
            # Calculate the moving weight
            weight_list = [i/float(sum(range(1,wma + 1))) for i in range(1,wma + 1)]
            
            for index, data in enumerate(raw_list):
                elements.append(float(data[5]))
                if len(elements) == wma:
                    raw_list[index].append(sum(np.multiply(weight_list,elements)))
                    del elements[0]
                else:
                    raw_list[index].append('')         
         
# Exponential smoothing           
        for es in es_forecast_rate:
            
            # Append a new header
            colheader.append('close-es' + str(round(es,2)))
            for index, data in enumerate(raw_list):
                if index == 0:
                    raw_list[index].append('')
                    # "forecast" record the pre forecast value
                    forecast = float(data[5])
                else:
                    raw_list[index].append((1-es) * ex_close + es * forecast)
                    forecast = (1-es) * float(ex_close) + es * forecast
                ex_close = float(data[5])            
                    
# Diff & MACD
        diff_elements = []
        colheader.append('di')
        
        for index, data in enumerate(raw_list):
            raw_list[index].append((float(data[3])+float(data[4])+2*float(data[5]))/4)
            #For MACD using
            diff_elements.append((float(data[3])+float(data[4])+2*float(data[5]))/4)
        #print (diff_elements)
            
# MACD
        # Pick the MACD list
        for content in macd_list:
            # The MACD can't work if the number of records is lower than the ma numbers.
            # Pick the MACD which is lower than the number of records
            if len(raw_list) > content:
                # new_ma_list is a list keeps proper number of ma
                new_macd_list.append(content)
        # MACD
        for macd in new_macd_list:
            elements = []
            colheader.append('close-macd' + str(macd))
            #print (diff_elements)
            for index, data in enumerate(diff_elements):
                #print (data)
                #elements.append(float(data[5]))
                elements.append(float(data))
                #print(elements)
                if len(elements) == macd:
                    # Append the MACD value to the botton of data list
                    raw_list[index].append(np.mean(elements))

                    # Delete the first close value of element
                    del elements[0]
                else:
                    raw_list[index].append('')
#        print("123")
# %K, %D, slow %D
        # Stochastic %K = (Ct - Ln)/(Hn - Ln) * 100
        # Stochastic %D = (%Kt + %Kt-1 + ... + %Kt-n+1)/n
        for stochastic in stochastic_interval:
            # Append a new header
            colheader.append('%K-' + str(stochastic))
            colheader.append('%D-' + str(stochastic))
            colheader.append('slow %D-' + str(stochastic))
            for index, data in enumerate(raw_list):
                if not int(data[6]) == 0 and not data[3] == data[4]:
                    high_list.append(float(data[3]))
                    low_list.append(float(data[4]))
                    if index < stochastic - 1:
                        raw_list[index].append('')
                        raw_list[index].append('')
                        raw_list[index].append('')
                    else:
                        max_value = max(high_list)
                        min_value = min(low_list)
                        # For debugging
                        #print(folder, split, index, max_value, min_value)
                        K_value = (float(data[5])-min_value)/(max_value-min_value)*100
                        raw_list[index].append(K_value)
                        D_list.append(K_value)
                        if len(D_list) == stochastic:
                            D_value = sum(D_list)/stochastic
                            raw_list[index].append(D_value)
                            del D_list[0]
                            slow_D_list.append(D_value)
                            if len(slow_D_list) == stochastic:
                                slow_D_value = sum(slow_D_list)/stochastic
                                raw_list[index].append(slow_D_value)
                                del slow_D_list[0]
                            else:
                                raw_list[index].append('')
                        else:
                            raw_list[index].append('')
                            raw_list[index].append('')
                        
                        del high_list[0]
                        del low_list[0]
                else:
                    raw_list[index].append('')
                    raw_list[index].append('')
                    raw_list[index].append('')      
# Momentum
        # Pick the momentum list
        for content in momentum_list:
            # The momentum can't work if the number of records is lower than the ma numbers.
            # Pick the momentum which is lower than the number of records
            if len(raw_list) > content:
                # new_ma_list is a list keeps proper number of ma
                new_momentum_list.append(content)
        # Momentum
        for mom in new_momentum_list:
            elements = []
            colheader.append('close-momentum' + str(mom))
            for index, data in enumerate(raw_list):
                elements.append(float(data[5]))
                if len(elements) == mom:
                    # Append the momentum value to the botton of data list
                    raw_list[index].append(float(elements[mom-1])-float(elements[0]))

                    # Delete the first close value of element
                    del elements[0]
                else:
                    raw_list[index].append('')
                    
# MA for Volume              
        # Pick the ma volume list
        ma_number =[]
        for content in ma_vol_list:
            # The MA can't work if the number of records is lower than the ma numbers.
            # Pick the ma which is lower than the number of records
            if len(raw_list) > content:
                # new_ma_list is a list keeps proper number of ma
                new_ma_vol_list.append(content)
                
        # Moving average (volume)
        for ma_vol in new_ma_list:
            # "elements" list contains ma number of data 
            elements = []
            # Append a new header
            colheader.append('volume_ma' + str(ma))

            
            for index, data in enumerate(raw_list):
                elements.append(float(data[6]))
                #ma_elements[0, index].append(float(data[5]))
                # If the "elements" list contains ma number of close value, then calculate the ma value (volume)
                if len(elements) == ma_vol:
                    # Append the ma value (volume) to the botton of data list
                    raw_list[index].append(np.mean(elements))
                    #ma_elements[1, index].append(np.mean(elements))

                    del elements[0]
                else:
                    raw_list[index].append('')
                    
# Ratio for Volume              

        for content in ratio_vol_list:
            # The ratio_vol can't work if the number of records is lower than the ma numbers.
            # Pick the ratio_vol which is lower than the number of records
            if len(raw_list) > content:
                # new_ratio_vol_list is a list keeps proper number of ma
                new_ratio_vol_list.append(content)
                
        # Moving average (volume)
        for ratio_vol in new_ma_list:
            # "elements" list contains ma number of data 
            elements = []
            # Append a new header
            colheader.append('volume_ratio' + str(ma))
            
            for index, data in enumerate(raw_list):
                elements.append(float(data[6]))
                #ma_elements[0, index].append(float(data[6]))
                # If the "elements" list contains ma number of close value, then calculate the ratio (volume)
                if len(elements) == ratio_vol:
                    # Append the ratio(volume) to the botton of data list
                    mean=np.mean(elements)
                    if mean > 0 :
                        raw_list[index].append(float(elements[ratio_vol-1])/mean)
                    else:
                        raw_list[index].append(raw_list[index-1])

                    del elements[0]
                else:
                    raw_list[index].append('')

        df = DataFrame(raw_list,columns = colheader)
        df.to_csv(new_Path + split, header=True, index=False)

        print(split + ' is done')

def main():
    for file in file_name:
        feature(file)

if __name__ == '__main__':
    main()
