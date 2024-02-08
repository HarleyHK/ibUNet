# ibUNet
The repository for the ibUNet project: A Lightweight Inception Boosted U-Net Neural Network for Routability Prediction

============================================================================================
The project is used to reproduce the experimental results shown in the paper: "A Lightweight Inception Boosted U-Net Neural
Network for Routability Prediction", submitted to the conference: ISEDA2024 (https://www.eda2.com/iseda/index.html).
It can run with the following software and hardware:
Windows 11 (Linux can also work with slight modification)
PyCharm 2023.3(Community Edition)
AMD (64bit) Ryzen 9 3900X 12-Core Processor, + 64G memory
GPU: NVIDIA GeForce GTX 3070, +8G GPU memory
9~10 hours (for 200,000 iterations)

============================================================================================

============================================================================================
Thanks a lot for the published dataset CircuitNet(https://github.com/circuitnet/CircuitNet).

(1) Prepare the data as advised in CircuitNet project.
Firstly, follow the tutorial on CircuitNet (https://circuitnet.github.io) to download the Routability Features;
Then, run the preprocessing script to generate the training set for corresponding tasks: Routing Congestion and DRC hotspot map predictions, (No IR Drop Prediction in our project!) 

(2) Set the two dataset paths for our two tasks:
As shown in our Windows version code, we can configure the two task dataset paths in utils/configs.py to:
"C:/Circuit_Net/congestion_DataSet/"
"C:/Circuit_Net/DRC_DataSet/"

(3) Batch run the code for reproducing the experimental results.
call the main function in the file: Run_Train_and_Inference.py with parameters: "--task Cong_or_DDR --xxx xxx"
Then about 20 hours later, we can get all the training and inference results for both tasks, as well as the test results for 20 epochs (1epoch==10,000 iterations).

============================================================================================

