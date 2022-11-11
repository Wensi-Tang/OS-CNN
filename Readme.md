!!!! Sometimes GitHub cannot render ipynb file. It's Github's [problem](https://github.com/jupyter/notebook/issues/3555#issuecomment-403361082) just wait for a few minutes and try again. !!!!



This is code for the paper [Omni-Scale CNNs: a simple and effective kernel size configuration for time series classification (ICLR 2022)](https://arxiv.org/abs/2002.10061)


	@inproceedings{tang2021omni,
	  title={Omni-Scale CNNs: a simple and effective kernel size configuration for time series classification},
	  author={Tang, Wensi and Long, Guodong and Liu, Lu and Zhou, Tianyi and Blumenstein, Michael and Jiang, Jing},
	  booktitle={International Conference on Learning Representations},
	  year={2021}
	}


### OS-CNN achieves SOTA on

[UCR and UEA](http://www.timeseriesclassification.com/) archives and some private datasets. 

#### with just the default hyperpermeter setting! no need to search!

Just have a try!!!


### Environment 

python == 3.5  
pytorch == 1.1.0  
scikit-learn == 0.21.3

### Easy use

Try [Google Colab](https://colab.research.google.com/)

Import this file [OS_CNN_Colab_demo.ipynb](https://github.com/Wensi-Tang/OS-CNN/blob/master/OS_CNN_Colab_demo.ipynb)

**or** 

Run With Jupyter Notebook

1\_1\_OS-CNN\_easy\_use\_Run\_and\_Save\_Model.ipynb

> This is an easy use of OS-CNN on univeriate dataset  
> Search `X_train, y_train, X_test, y_test = TSC_data_loader(dataset_path, dataset_name)`  
> you could replace the `X_train, y_train, X_test, y_test` as you want, or you could change dataset_name to determine which UCR dataset you want to run


2\_1\_1\_OS-CNN\_easy\_use\_Run\_and\_Save\_Model\_for\_multivariate.ipynb

> This is an easy use OS-CNN on multivariate dataset  
> search `X_train, y_train, X_test, y_test = TSC_multivariate_data_loader(dataset_path, dataset_name)`  
> you could replace the `X_train, y_train, X_test, y_test` as you want, or you could change dataset_name to determine which UEA dataset you want to run

### Full Results

In ./Full_Results folder
We have results of OS-CNN for UCR 85 datasets, UCR 128 datasets, and UEA 30 datasets.




### I cannot see anything ###
Github some times cannot render ipynb file if you find some pages cannot load plz wait for a while, and try again. See [this](https://github.com/jupyter/notebook/issues/3555#issuecomment-403361082)




### Detailed description of each file
1\_1\_OS-CNN\_easy\_use\_Run\_and\_Save\_Model.ipynb  
>  This is an easy use OS-CNN  
>  search `X_train, y_train, X_test, y_test = TSC_data_loader(dataset_path, dataset_name)`  
>   you could replace the `X_train, y_train, X_test, y_test` as you like, or you could change dataset_name to determine which UCR dataset you want to run


1\_2\_OS-CNN\_load\_saved\_model\_for\_prediction.ipynb   
> This code could help you to load morel and use the model for prediction (it needs model trained by 1\_1\_OS-CNN\_easy\_use\_Run\_and\_Save\_Model.ipynb)


2\_1\_1\_OS-CNN\_easy\_use\_Run\_and\_Save\_Model\_for\_multivariate.ipynb
>  This is an easy use OS-CNN on multivariate dataset  
>  search `X_train, y_train, X_test, y_test = TSC_multivariate_data_loader(dataset_path, dataset_name)`  
>  you could replace the `X_train, y_train, X_test, y_test` as you like, or you could change dataset_name to determine which UEA dataset you want to run


2\_2\_1\_OS\_OS-CNN\_easy\_use\_Run\_and\_Save\_Model\_for\_multivariate.ipynb
>  This is an easy use OS\_OS-CNN on multivariate dataset  
>  the OS\_OS-CNN is using OS layer on each variate of multivariate then put the feature map into an OS-CNN  
>  search `X_train, y_train, X_test, y_test = TSC_multivariate_data_loader(dataset_path, dataset_name)`  
>  you could replace the `X_train, y_train, X_test, y_test` as you like, or you could change dataset_name to determine which UEA dataset you want to run


3\_1\_compare\_result.ipynb
>  In here, you could select different models to compare with os-cnn


Folder ./Code\_example\_of\_theoretical\_proof/ has the code verification of theoretical proof for our paper


    1_1_Deep_Learning_Convolution_and_Convolution_theorem.ipynb
        Code verification of Section 3.2
    
    2_1_Time_and_Space_Complexity_of_OS-CNN_Vs_FCN_ResNet.ipynb
        This code shows the model size of OS-CNN and Resnet and FCN. 
        It shows the OS-CNN is of better time and space complexity than SOTA
    
    3_1_verification _of_Pytorch_FCN_&_ResNet_implementation.ipynb
        This code verifies the FCN and ResNet Pytorch implementation is correct
        
    3_2_FCN_with_different_kernel_size.ipynb
        This code gets the classification result of FCN with different kernel sizes. Section 6.2 Table 3
    
    3_3_Positional_information_loss_of_FCN_and_how_OS-CNN_overcome_this.ipynb
        This code shows the positional information loss of fixed kernel size design. Section 3.4
    
    4_1_OS-CNN_load_saved_model_and_visualization_weight.ipynb 
        Check the initial noise and its influence on the feature extraction. Section 3.4
        
    4_2_Frequency_Resolution.ipynb
        Check frequency resolution of small kernel size. Section 3.4
        
    4_3_Check_Capability_Equivalent.ipynb
        This is code for Section 5: No representation ability lose 
        
    4_4_calculate_prime_model_size.ipynb
        This is code for Section 5: Smaller model size
        
    4_5_Enough_channel.ipynb
        This is code for Section 5.

Folder ./Appendix has some supplementary material:


	1. Proof of No representation ability lose is a theoretical proof of no representation ability lose
	2. The novelty of OS-CNN is a demonstration for why it can reduce model size
	3. OS-CNN_network_structure.ipynb  It shows the network structure of OS-CNN


​    
Folder ./Texas\_Sharpshooter\_plot has materials for comparison between OS-CNN and cDTW by Texas Sharpshooter plot   
​        
​    
