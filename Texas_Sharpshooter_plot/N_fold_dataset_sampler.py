import numpy as np

class N_fold_dataset_sampler():
    def __init__(self, y_label,N_fold, shuffle = False, random_seed = None):
        super(N_fold_dataset_sampler, self).__init__()
        if random_seed!= None:
            np.random.seed(random_seed)
        self.shuffle = shuffle
        self.y_label = y_label
        self.N_fold = N_fold
        self.index_info_for_each_class = self.get_total_number_and_index_of_instance_in_each_class(self.y_label)
        

                 
    def get_total_number_and_index_of_instance_in_each_class(self, y_label):
        index_info_for_each_class = []
        class_number = np.max(y_label)+1 
        for ith_class in range(class_number):
            index_list_of_ith_class = np.where(self.y_label==ith_class)[0]
            if self.shuffle:
                np.random.shuffle(index_list_of_ith_class)
            total_number_of_instance_of_ith_class = index_list_of_ith_class.shape[0]
            index_info_for_each_class.append([ith_class, total_number_of_instance_of_ith_class, index_list_of_ith_class])
        return index_info_for_each_class
    
    def __iter__(self):
        self.Nth_iter = 0
        return self
    
    def __next__(self):
        if self.Nth_iter >=self.N_fold:
            raise StopIteration
        else:
            self.Nth_iter = self.Nth_iter + 1
            
            sampled_index, lefted_index = self.select_index()
            
            return np.asarray(lefted_index), np.asarray(sampled_index)
        
    def select_index(self):
        sampled_index_list =[]
        lefted_index_list = []
        for index_info in self.index_info_for_each_class:
            ith_class = index_info[0]
            total_number_of_instance_of_ith_class = index_info[1]
            index_list_of_ith_class = index_info[2]
            if total_number_of_instance_of_ith_class>=self.N_fold:
                sampled_index = np.array_split(index_list_of_ith_class, self.N_fold)[self.Nth_iter-1]
                lefted_index = index_list_of_ith_class[~np.isin(index_list_of_ith_class,sampled_index)]
            elif total_number_of_instance_of_ith_class==1:
                sampled_index = index_list_of_ith_class
                lefted_index = index_list_of_ith_class
            else:
                sampled_index = index_list_of_ith_class[(self.Nth_iter-1)%total_number_of_instance_of_ith_class]
                lefted_index = index_list_of_ith_class[~np.isin(index_list_of_ith_class,sampled_index)]
                

            if sampled_index.size == 1:
                sampled_index_list.append(np.asscalar(sampled_index))
            else:
                for i in sampled_index:
                    sampled_index_list.append(i)
            if lefted_index.size == 1:
                lefted_index_list.append(np.asscalar(lefted_index))
            else:
                for i in lefted_index:
                    lefted_index_list.append(i)
            
        return sampled_index_list, lefted_index_list