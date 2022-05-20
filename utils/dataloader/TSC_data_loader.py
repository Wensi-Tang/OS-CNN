
from sklearn import preprocessing
from PIL import Image
from sklearn.preprocessing import minmax_scale
import numpy as np

def set_nan_to_zero(a):
    where_are_NaNs = np.isnan(a)
    a[where_are_NaNs] = 0
    return a


def check_dataset(sorted_label_data):
    #check sort value and check number of each number
    label = sorted_label_data[:,0:1]
    Nor_data = minmax_scale(sorted_label_data[:,1:])
    Nor_label = minmax_scale(sorted_label_data[:,0:1])
    print(Nor_label.shape)
    biglabel = np.repeat(Nor_label, 80, axis=1)
    big = np.concatenate((biglabel, Nor_data),axis=1)
    img = Image.fromarray(big*255)
    img.show()
    unique, counts = np.unique(label, return_counts=True)
    print(dict(zip(unique, counts)))

def trim_lable(dataset):
    data = dataset[:,1:]
    label = dataset[:,0:1]
    le = preprocessing.LabelEncoder()
    le.fit(np.squeeze(label, axis=1))
    label = le.transform(np.squeeze(label, axis=1))
    label = np.expand_dims(label,axis =1)
    dataset = np.concatenate((label, data), axis=1)
    return dataset

def split_by_class(sorted_label_data,train_list):
    train_index = []
    test_index = []
    for i in range(sorted_label_data.shape[0]):
        if sorted_label_data[i,0] in train_list:
            train_index.append(i)
        else:
            test_index.append(i)
    train_dataset = sorted_label_data[train_index,:]
    test_dataset = sorted_label_data[test_index,:]

    return trim_lable(train_dataset), trim_lable(test_dataset)


def normal_datset_to_few_shot_dataset(X_train, y_train, X_test, y_test,train_ratio=0.8, seed=None):
    #biuld big dataset

    data = np.concatenate((X_train,X_test))
    label = np.concatenate((y_train,y_test))
    label = np.expand_dims(label,axis =1 )
    label_data = np.concatenate((label, data),axis=1)

    # sample classes
    n_class = np.amax(label)+1
    sorted_label_data = label_data[label[:, 0].argsort()]
    if seed == None:
        list = np.random.choice(n_class, n_class, replace=False)
    else:
        np.random.seed(seed=seed)
        list = np.random.choice(n_class, n_class, replace=False)
    train_list = list[0:int(n_class*train_ratio)]

    # check_dataset(sorted_label_data)

    #split dataset
    train_dataset, test_dataset = split_by_class(sorted_label_data, train_list)

    return train_dataset, test_dataset, train_list

def normal_datset_to_few_shot_dataset_with_list(X_train, y_train, X_test, y_test,train_list):

    data = np.concatenate((X_train,X_test))
    label = np.concatenate((y_train,y_test))
    label = np.expand_dims(label,axis =1 )
    label_data = np.concatenate((label, data),axis=1)

    sorted_label_data = label_data[label[:, 0].argsort()]

    train_dataset, test_dataset = split_by_class(sorted_label_data, train_list)

    return train_dataset, test_dataset, train_list

def fill_out_with_Nan(data,max_length):
    #via this it can works on more dimensional array
    pad_length = max_length-data.shape[-1]
    if pad_length == 0:
        return data
    else:
        pad_shape = list(data.shape[:-1])
        pad_shape.append(pad_length)
        Nan_pad = np.empty(pad_shape)*np.nan
        return np.concatenate((data, Nan_pad), axis=-1)
    

def get_label_dict(file_path):
    label_dict ={}
    with open(file_path) as file:
        lines = file.readlines()
        for line in lines:
            if '@classLabel' in line:
                label_list = line.replace('\n','').split(' ')[2:]
                for i in range(len(label_list)):
                    label_dict[label_list[i]] = i 
                
                break
    return label_dict


def get_data_and_label_from_ts_file(file_path,label_dict):
    with open(file_path) as file:
        lines = file.readlines()
        Start_reading_data = False
        Label_list = []
        Data_list = []
        max_length = 0
        for line in lines:
            if Start_reading_data == False:
                if '@data'in line:
                    Start_reading_data = True
            else:
                temp = line.split(':')
                Label_list.append(label_dict[temp[-1].replace('\n','')])
                data_tuple= [np.expand_dims(np.fromstring(channel, sep=','), axis=0) for channel in temp[:-1]]
                max_channel_length = 0
                for channel_data in data_tuple:
                    if channel_data.shape[-1]>max_channel_length:
                        max_channel_length = channel_data.shape[-1]
                data_tuple = [fill_out_with_Nan(data,max_channel_length) for data in data_tuple]
                data = np.expand_dims(np.concatenate(data_tuple, axis=0), axis=0)
                Data_list.append(data)
                if max_channel_length>max_length:
                    max_length = max_channel_length
        
        Data_list = [fill_out_with_Nan(data,max_length) for data in Data_list]
        X =  np.concatenate(Data_list, axis=0)
        Y =  np.asarray(Label_list)
        
        return np.float32(X), Y

    
import scipy.io as sio

def get_from_X(X):
    data_list = []
    max_length = 0
    for data in X[0][0][0][:]:
        data = np.expand_dims(data,0)
        data_list.append(data)
        max_channel_length = data.shape[-1]
        if max_channel_length>max_length:
            max_length = max_channel_length        
    Data_list = [fill_out_with_Nan(data,max_length) for data in data_list]
    X =  np.concatenate(Data_list, axis=0)
    return np.float32(X)

def get_from_Y(y):
    y = y[0][0].flatten()
    return np.int64(y)



def TSC_multivariate_data_loader_from_mat(dataset_path, dataset_name):
    full_path = dataset_path+'/'+dataset_name+'/'+dataset_name+'.mat'
    mat_contents = sio.loadmat(full_path)
    X_train_raw = mat_contents['mts']['train']
    y_train_raw = mat_contents['mts']['trainlabels']
    X_test_raw =mat_contents['mts']['test']
    y_test_raw = mat_contents['mts']['testlabels']
    X_train = get_from_X(X_train_raw)
    y_train = get_from_Y(y_train_raw)
    X_test = get_from_X(X_test_raw)
    y_test = get_from_Y(y_test_raw)
    le = preprocessing.LabelEncoder()
    le.fit(y_train)
    y_train  = le.transform(y_train)
    y_test  = le.transform(y_test)
    return set_nan_to_zero(X_train), y_train, set_nan_to_zero(X_test), y_test




def TSC_multivariate_data_loader(dataset_path, dataset_name):
    
    Train_dataset_path = dataset_path + '/' + dataset_name + '/' + dataset_name + '_TRAIN.ts'
    Test_dataset_path = dataset_path + '/' + dataset_name + '/' + dataset_name + '_TEST.ts'
    label_dict = get_label_dict(Train_dataset_path)
    X_train, y_train = get_data_and_label_from_ts_file(Train_dataset_path,label_dict)
    X_test, y_test = get_data_and_label_from_ts_file(Test_dataset_path,label_dict)
    
    return set_nan_to_zero(X_train), y_train, set_nan_to_zero(X_test), y_test


def TSC_data_loader(dataset_path,dataset_name):
    Train_dataset = np.loadtxt(
        dataset_path + '/' + dataset_name + '/' + dataset_name + '_TRAIN.tsv')
    Test_dataset = np.loadtxt(
        dataset_path + '/' + dataset_name + '/' + dataset_name + '_TEST.tsv')
    Train_dataset = Train_dataset.astype(np.float32)
    Test_dataset = Test_dataset.astype(np.float32)

    X_train = Train_dataset[:, 1:]
    y_train = Train_dataset[:, 0:1]

    X_test = Test_dataset[:, 1:]
    y_test = Test_dataset[:, 0:1]
    le = preprocessing.LabelEncoder()
    le.fit(np.squeeze(y_train, axis=1))
    y_train = le.transform(np.squeeze(y_train, axis=1))
    y_test = le.transform(np.squeeze(y_test, axis=1))
    return set_nan_to_zero(X_train), y_train, set_nan_to_zero(X_test), y_test


def check_normalized(X_train,X_test,dataset_name):
    mean_of_feature_cols_train = np.nanmean(X_train, axis=1, keepdims= True)
    std_of_feature_cols_train = np.nanstd(X_train, axis=1, keepdims= True)
    if np.nanmean(abs(mean_of_feature_cols_train)) < 1e-7 and abs(np.nanmean(std_of_feature_cols_train)-1) < 0.05 :
        return X_train,X_test
    else:
        print(dataset_name,"is not normalized, let's do it")
        print('mean = ',np.nanmean(mean_of_feature_cols_train), 'std = ',np.nanmean(std_of_feature_cols_train))
        mean_of_feature_cols_test = np.nanmean(X_test, axis=1, keepdims= True)
        std_of_feature_cols_train = np.nanstd(X_train, axis=1, keepdims= True)
        std_of_feature_cols_test = np.nanstd(X_test, axis=1, keepdims= True)
        X_train = (X_train -mean_of_feature_cols_train)/std_of_feature_cols_train
        X_test = (X_test -mean_of_feature_cols_test)/std_of_feature_cols_test
        return X_train, X_test
    

def TSC_data_loader_with_z_normaliz_check(dataset_path,dataset_name):
    Train_dataset = np.loadtxt(
        dataset_path + '/' + dataset_name + '/' + dataset_name + '_TRAIN.tsv')
    Test_dataset = np.loadtxt(
        dataset_path + '/' + dataset_name + '/' + dataset_name + '_TEST.tsv')
    Train_dataset = Train_dataset.astype(np.float32)
    Test_dataset = Test_dataset.astype(np.float32)

    X_train = Train_dataset[:, 1:]
    y_train = Train_dataset[:, 0:1]

    X_test = Test_dataset[:, 1:]
    y_test = Test_dataset[:, 0:1]
    le = preprocessing.LabelEncoder()
    le.fit(np.squeeze(y_train, axis=1))
    y_train = le.transform(np.squeeze(y_train, axis=1))
    y_test = le.transform(np.squeeze(y_test, axis=1))
    
    
    X_train,X_test = check_normalized(X_train,X_test,dataset_name)
    
    return set_nan_to_zero(X_train), y_train, set_nan_to_zero(X_test), y_test
