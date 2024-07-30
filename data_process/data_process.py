import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

#about 10 data is 1 second
data_in_epoch = 40

def low_pass_filter(data, alpha=0.1):
    filtered_data = np.zeros_like(data)  # Initialize an array of the same shape as data
    filtered_data[0] = data[0]  # Set the first value to be the same as the first data point
    for i in range(1, len(data)):  # Iterate over the rest of the data points
        filtered_data[i] = alpha * data[i] + (1 - alpha) * filtered_data[i-1]
    return filtered_data

def abstract_attributes(filepath):
    original_data = pd.read_csv(filepath, header=None)
    original_data = original_data.to_numpy()
    #abstract each line of pure data
    accel_data = []
    for data in original_data[:,2]:
        str_list = data.split(' ')[-1]
        list = eval(str_list)
        accel_data.append(list)
    #slice data into epochs; abstract attrbutes from first 3 epochs; store each epochs
    attributes_list = []
    for point in accel_data:
        data_slice = np.array(point)
        #number data in each line should be 165, skip the line if not
        if data_slice.size != 495:
            continue
        else:
            #only use the first three epoch in one line of data
            for i in range(0,3):
                data_point = data_slice[(data_in_epoch*i):(data_in_epoch*(i+1))]
                mean_x = (data_point[:,0].mean())
                mean_y = (data_point[:,1].mean())
                mean_z = (data_point[:,2].mean())
                sum_x = (data_point[:,0].sum())
                sum_y = (data_point[:,1].sum())
                sum_z = (data_point[:,2].sum())    
                min_x = (data_point[:,0].min())    
                min_y = (data_point[:,1].min())  
                min_z = (data_point[:,2].min())
                max_x = (data_point[:,0].max())    
                max_y = (data_point[:,1].max())  
                max_z = (data_point[:,2].max())  
                sd_x = (np.std(data_point[:,0]))
                sd_y = (np.std(data_point[:,1]))
                sd_z = (np.std(data_point[:,2]))
                skew_x = (skew(data_point[:,0]))
                skew_y = (skew(data_point[:,1]))
                skew_z = (skew(data_point[:,2]))
                kurt_x = (kurtosis(data_point[:,0]))
                kurt_y = (kurtosis(data_point[:,1]))
                kurt_z = (kurtosis(data_point[:,2]))
                corre_xy = np.corrcoef(data_point[:,0], data_point[:,1])[0,1]
                corre_xz = np.corrcoef(data_point[:,0], data_point[:,2])[0,1]
                corre_yz = np.corrcoef(data_point[:,1], data_point[:,2])[0,1]
                vm_mean = np.linalg.norm([mean_x,mean_y,mean_z])
                vm_sum = np.linalg.norm([sum_x,sum_y,sum_z])
                vm_min = np.linalg.norm([min_x,min_y,min_z])
                vm_max = np.linalg.norm([max_x,max_y,max_z])
                vm_sd = np.linalg.norm([sd_x,sd_y,sd_z])
                vm_kurt = np.linalg.norm([kurt_x,kurt_y,kurt_z])
                vm_skew = np.linalg.norm([skew_x,skew_y,skew_z])
                #measure ODBA
                static_acc_x = low_pass_filter(data_point[:, 0])
                static_acc_y = low_pass_filter(data_point[:, 1])
                static_acc_z = low_pass_filter(data_point[:, 2])
                static_acc = np.vstack((static_acc_x, static_acc_y, static_acc_z)).T  # Combine the filtered data
                dynamic_acc = data_point - static_acc
                ODBA = np.sum(np.abs(dynamic_acc), axis=1).sum()

                attributes = [mean_x,mean_y,mean_z,sum_x,sum_y,sum_z,min_x,min_y,min_z,max_x,max_y,max_z,sd_x,sd_y,sd_z,skew_x,skew_y,skew_z,kurt_x,kurt_y,kurt_z,corre_xy,corre_xz,corre_yz,vm_mean,vm_sum,vm_min,vm_max,vm_sd,vm_kurt,vm_skew, ODBA]
                attributes_list.append(attributes)
    return attributes_list

if __name__ == "__main__":
    attributes_list = abstract_attributes("./original_data/6.5.csv") + abstract_attributes("./original_data/6.6.csv") + abstract_attributes("./original_data/6.7.csv")
    label = [['mean_x','mean_y','mean_z','sum_x','sum_y','sum_z','min_x','min_y','min_z','max_x','max_y','max_z','sd_x','sd_y','sd_z','skew_x','skew_y','skew_z','kurt_x','kurt_y','kurt_z','corre_xy','corre_xz','corre_yz','vm_mean','vm_sum','vm_min','vm_max','vm_sd','vm_kurt','vm_skew', 'ODBA']]
    label = np.array(label)
    attributes_list = np.array(attributes_list)
    result = np.vstack((label, attributes_list), dtype="object")
    np.savetxt("./attributes_data.csv", result, delimiter=',', fmt="%s")
    print(f"Process complete, save as attributes_data.csv. One attributes list represent {data_in_epoch} XYZ accel datapoint.")
