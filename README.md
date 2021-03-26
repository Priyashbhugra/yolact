# yolact

 This is the code for our papers: - [YOLACT: Real-time Instance Segmentation](https://arxiv.org/abs/1904.02689)




This repo will help you detect and segment the instances of the cars: 
Below you can see the results from this YOLACT model.

![Example 0](data/yolact_example_0.png)

![Example 1](data/yolact_example_1.png)

![Example 2](data/yolact_example_2.png)

# Installation
 - Clone this repository and enter it:
   ```Shell
   git clone https://github.com/Priyashbhugra/yolact.git
   cd yolact
   ```
 - Create Anaconda environment:
   ```Shell
   conda create -n yolact 
  
      ```

 - Run the below command and wait untill the envionment is created:
   ```Shell
   conda env create -f environment.yml`
   ```


# Evaluation
Here are our YOLACT models pretrained on COCO dataset:

- Download Resnet101-FPN ----------[yolact_im700_54_800000.pth](https://drive.google.com/file/d/1lE4Lz5p25teiXV-6HdTiOJSnS7u7GBzg/view?usp=sharing)

I am only using Resnet101-FPN as a backbone.
To evalute the model, put the corresponding weights file in the `weights` directory and run one of the following commands.

## Results on pretrained COCO dataset
- Run the below command for evaluation
```Shell
python eval.py --trained_model=weights/yolact_im700_54_800000.pth --score_threshold=0.15 --image=car.jpeg --cross_class_nms=True
```
