from __future__ import print_function

from inference import inference
from train import train
import sys


def work_task(task_name,LogFileName):
    print('This message will be displayed on the screen.')
    original_stdout = sys.stdout  # Save a reference to the original standard output
    with open(LogFileName, 'w') as f:
        sys.stdout = f  # Change the standard output to the file we created.
        print('This message will be written to a file.')
        # #////////////////////////////////////////////////////
        # # Training:
        _command, _task, drc_cong, _Pretrained, model_path = sys.argv[0:5]
        sys.argv[1] = "--task"
        sys.argv[2] = task_name
        sys.argv[3] = "--save_path"
        sys.argv[4] ="./"+task_name+"/"
        print("Run Training...")
        train()
        # # ////////////////////////////////////////////////////

        # "--task Congestion --save_path ./Congestion_Models/"
        modelList= [
                    'model_iters_10000.pth',
                    'model_iters_20000.pth',
                    'model_iters_30000.pth',
                    'model_iters_40000.pth',
                    'model_iters_50000.pth',
                    'model_iters_60000.pth',
                    'model_iters_70000.pth',
                    'model_iters_80000.pth',
                    'model_iters_90000.pth',
                    'model_iters_100000.pth',
                    'model_iters_110000.pth',
                    'model_iters_120000.pth',
                    'model_iters_130000.pth',
                    'model_iters_140000.pth',
                    'model_iters_150000.pth',
                    'model_iters_160000.pth',
                    'model_iters_170000.pth',
                    'model_iters_180000.pth',
                    'model_iters_190000.pth',
                    'model_iters_200000.pth',
                    ]

        #////////////////////////////////////////////////////
        # "--task DRC --pretrained .\DRC_Models\model_iters_10000.pth"
        # Inference:
        sys.argv[1] = "--task"
        sys.argv[2] = task_name
        sys.argv[3] = "--pretrained"
        _command, _task, drc_cong, _Pretrained, model_path = sys.argv[0:5]
        print("Run Inference...")
        # ////////////////////////////////////////////////////

        if(len(sys.argv)>=7):
            sys.argv[7] = sys.argv[7]+"_"+task_name

        for kk in range(len(modelList)):
            model_path = modelList[kk]
            sys.argv[4]=".\\"+ drc_cong+"\\"+ model_path
            print("Model: ["+model_path+"] Inference...")
            inference()

        # ============================================================================
        print('This message will be displayed back to the screen.')
        sys.stdout = original_stdout  # Reset the standard output to its original value

# "--task Cong_or_DR --xxx xxx"
# "--task Cong_or_DRC --xxx xxx --plot_roc  --save_path ROC_Folder" for plot ROC curve.
if __name__ == "__main__":
    work_task("Congestion", 'Congestion_LogFile.txt')
    work_task("DRC", 'DRC_LogFile.txt')


