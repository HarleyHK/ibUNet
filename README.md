# ibUNet

The following information regarding the ibUNet project's repository, which is used to reproduce the experimental results presented in the paper titled "A Lightweight Inception Boosted U-Net Neural Network for Routability Prediction" submitted to the conference ISEDA2024 (https://www.eda2.com/iseda/index.html):

The project requires the following software and hardware to run effectively: Windows 11 (Linux may also work with slight modifications), PyCharm 2023.3 (Community Edition), AMD (64bit) Ryzen 9 3900X 12-Core Processor, + 64G memory, and GPU: NVIDIA GeForce GTX 3070, +8G GPU memory. The project takes around 9-10 hours to run on one task for 200,000 iterations.

We would also like to express our gratitude for the published dataset CircuitNet (https://github.com/circuitnet/CircuitNet). 
(1) To prepare the data, please follow the guidelines stated in the CircuitNet project. Firstly, download the Routability Features by following the tutorial on CircuitNet (https://circuitnet.github.io). Then, run the preprocessing script to generate the training set for the corresponding tasks: Routing Congestion and DRC hotspot map predictions (No IR Drop Prediction in our project!).

(2) To set the two dataset paths for our two tasks, the following steps should be taken: as shown in our Windows version code, configure the two task dataset paths in utils/configs.py as "C:/Circuit_Net/congestion_DataSet/" and "C:/Circuit_Net/DRC_DataSet/".

(3) To reproduce the experimental results, batch run the code by calling the main function in the file Run_Train_and_Inference.py with parameters "--task Cong_or_DDR --xxx xxx". After approximately 20 hours, all the training and inference results for both tasks, as well as the test results for 20 epochs (1epoch==10,000 iterations), will be available.

