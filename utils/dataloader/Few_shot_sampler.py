import numpy as np

class Few_shot_sampler(object):
    def __init__(self, dataset, m_way_n_shot_q_quary, iterations):
        #self.data = dataset[:,1:]
        self.lable = np.squeeze(dataset[:,0:1], axis=1)
        self.m_way = m_way_n_shot_q_quary[0]
        self.n_shot = m_way_n_shot_q_quary[1]
        self.q_quary = m_way_n_shot_q_quary[2]
        self.n_class = int(np.amax(self.lable)+1)
        self.iterations = iterations
        self.index_of_class = self.creat_dict_of_class_and_list(self.lable)
    def __iter__(self):
        for it in range(self.iterations):
            choosed_class = np.random.choice(self.n_class, self.m_way, replace=False)
            support_list = []
            query_list = []
            #sampled_list = []
            for number in choosed_class:
                temp_list = self.index_of_class[number]
                index_list = np.random.choice(temp_list, self.n_shot+self.q_quary, replace=False)
                support_list.extend(index_list[0:self.n_shot])
                query_list.extend(index_list[self.n_shot:])
                #sampled_list.extend(index_list)
            #yield sampled_list, support_list, query_list
            yield support_list, query_list
            #yield sampled_list
    def creat_dict_of_class_and_list(self,lable):
        dict = {}
        for i in range(self.n_class):
            dict[i] = []
        #print(dict)
        for i in range(lable.shape[0]):
            dict[int(lable[i])].append(i)
        return dict