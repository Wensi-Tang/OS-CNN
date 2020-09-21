This is code for the paper here: https://arxiv.org/abs/2002.10061


###
For use： 
my environment is:
python == 3.5
pytorch == 1.1.0
scikit-learn == 0.21.3.

### Easy use ###
1_1_OS-CNN_easy_use_Run_and_Save_Model.ipynb
    This is a easy use OS-CNN
    search "X_train, y_train, X_test, y_test = TSC_data_loader(dataset_path, dataset_name)"
    you could replace the "X_train, y_train, X_test, y_test" as you like, or you could change dataset_name to determine which UCR dataset you want to run
###


### I cannot see anything ###
Github some times cannot render ipynb file if you find some pages cannot load plz wait for a while, and try again!!!!
see this
https://github.com/jupyter/notebook/issues/3555#issuecomment-403361082
!!!!!
####


#### read me ####


1_1_OS-CNN_easy_use_Run_and_Save_Model.ipynb
    This is a easy use OS-CNN
    search "X_train, y_train, X_test, y_test = TSC_data_loader(dataset_path, dataset_name)"
    you could replace the "X_train, y_train, X_test, y_test" as you like, or you could change dataset_name to determine which UCR dataset you want to run

    
1_2_OS-CNN_load_saved_model_for_prediction.ipynb
    This code could help you to load morel and use the model for prediction (it should be used after 1_1 both them can train and save model)
    
1_3_OS-CNN_network_structure.ipynb
    This shows the network structure of OS-CNN

1_4_compare_result.ipynb
    In here, you could select different models to compare with os-cnn


Folder ./Code_example_of_theoretical_proof/ has the code verification of theoretical proof for our paper


    1_1_Deep_Learning_Convolution_and_Convolution_theorem.ipynb
        Code verification of Section 3.2
    
    2_1_Time_and_Space_Complexity_of_OS-CNN_Vs_FCN_ResNet.ipynb
        This code shows the model size of OS-CNN and Resnet and FCN. It shows the OS-CNN is of better time and space complexity than SOTA
    
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

Appendix folder is some supplementary material:
	1. Proof of No representation ability lose is a theoretical proof of no representation ability lose
	2. The novelty of OS-CNN is a demonstration for why it can reduce model size.
        
    