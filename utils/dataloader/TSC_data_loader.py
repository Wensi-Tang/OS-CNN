
from sklearn import preprocessing
from PIL import Image
from sklearn.preprocessing import minmax_scale
import numpy as np

def set_nan_to_zero(a):
    where_are_NaNs = np.isnan(a)
    a[where_are_NaNs] = 0
    return a

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
