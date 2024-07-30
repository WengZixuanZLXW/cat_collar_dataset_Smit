import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

#about 20 data is 1 seconds
data_in_epoch = 60

def abstract_attributes(filepath):
    original_data = pd.read_csv(filepath, header=None)
    original_data = original_data.to_numpy()
    #abstract each line of pure data
    accel_data = []
    for data in original_data:
        str_list = data[2].split(' ')[-1]
        list = eval(str_list)
        list.append(data[1])        
        accel_data.append(list)
    #slice data into epochs; abstract attrbutes from first 3 epochs; store each epochs
    attributes_list = []
    for point in accel_data:
        nppoint = np.array(point[0:165])   
        #number data in each line should be 165, skip the line if not
        if len(nppoint) != 165:
            continue
        else:
            for i in range(0,2):
                data_slice = nppoint[(data_in_epoch*i):(data_in_epoch*(i+1))]
                data_slice = data_slice/250   
                # the XYZ columns are based on the https://www.mdpi.com/1424-8220/23/16/7165
                mean_x = (data_slice[:,0].mean())
                min_x = (data_slice[:,0].min())  
                max_x = (data_slice[:,0].max())
                sd_x = (np.std(data_slice[:,0]))
                skew_x = (skew(data_slice[:,0]))
                kurt_x = (kurtosis(data_slice[:,0]))

                mean_y = (data_slice[:,1].mean())
                min_y = (data_slice[:,1].min())  
                max_y = (data_slice[:,1].max())  
                sd_y = (np.std(data_slice[:,1]))
                skew_y = (skew(data_slice[:,1]))
                kurt_y = (kurtosis(data_slice[:,1]))

                mean_z = (data_slice[:,2].mean())  
                min_z = (data_slice[:,2].min())
                max_z = (data_slice[:,2].max())  
                sd_z = (np.std(data_slice[:,2]))
                skew_z = (skew(data_slice[:,2]))
                kurt_z = (kurtosis(data_slice[:,2]))

                corre_xy = np.corrcoef(data_slice[:,0], data_slice[:,1])[0,1]
                corre_xz = np.corrcoef(data_slice[:,0], data_slice[:,2])[0,1]
                corre_yz = np.corrcoef(data_slice[:,2], data_slice[:,1])[0,1]

                vm_mean = np.linalg.norm([mean_x,mean_y,mean_z])
                vm_min = np.linalg.norm([min_x,min_y,min_z])
                vm_max = np.linalg.norm([max_x,max_y,max_z])
                vm_sd = np.linalg.norm([sd_x,sd_y,sd_z])
                vm_skew = np.linalg.norm([skew_x,skew_y,skew_z])
                vm_kurt = np.linalg.norm([kurt_x,kurt_y,kurt_z])

                attributes = [mean_x, min_x, max_x, sd_x, skew_x, kurt_x, mean_y, min_y, max_y, sd_y, skew_y, kurt_y, mean_z, min_z, max_z, sd_z, skew_z, kurt_z, vm_mean, vm_min, vm_max, vm_sd, vm_skew, vm_kurt, corre_xy, corre_xz, corre_yz, point[-1]]
                #attributes = [sd_x, skew_x, kurt_x, sd_y, skew_y, kurt_y, sd_z, skew_z, kurt_z, vm_sd, vm_skew, vm_kurt, corre_xy, corre_xz, corre_yz, point[-1]]
                
                attributes_list.append(attributes)
    return attributes_list

if __name__ == "__main__":
    attributes_list = abstract_attributes("./original_data/eat.csv")
    label = [['X_Mean', 'X_Min', 'X_Max', 'X_sd', 'X_Skew', 'X_Kurt',
       'Y_Mean', 'Y_Min', 'Y_Max', 'Y_sd', 'Y_Skew', 'Y_Kurt', 'Z_Mean',
       'Z_Min', 'Z_Max', 'Z_sd', 'Z_Skew', 'Z_Kurt', 'VM_Mean', 'VM_Min',
       'VM_Max', 'VM_sd', 'VM_Skew', 'VM_Kurt', 'Cor_XY', 'Cor_XZ', 'Cor_YZ',
       'Class']]
    label = np.array(label)
    attributes_list = np.array(attributes_list)
    result = np.vstack((label, attributes_list), dtype="object")
    np.savetxt("./data/valid.csv", result, delimiter=',', fmt="%s")
    print(f"Process complete, save as valid.csv. One attributes list represent {165} XYZ accel datapoint.")