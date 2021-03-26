# yolact

 This is the code from the papers: - [YOLACT: Real-time Instance Segmentation](https://arxiv.org/abs/1904.02689)
 
 I have used and updated the code from https://github.com/dbolya/yolact.git



This repo will help you detect and segment the instances of the cars: 
Below you can see the results from this YOLACT model.

Here we tried to detect and genarate a mask for all the cars in an image.

![Example 0](many_objects_results.png)

![Example 1](cars_with_pededtrian_1.png)

![Example 2](car_result.png)

# Installation
 - Clone this repository and enter it:
   ```Shell
   git clone https://github.com/Priyashbhugra/yolact.git
   cd yolact
   ```
 - Create Anaconda environment:
   ```Shell
   conda create -n yolact
   conda activate yolact
  
      ```

 - Run the below command and wait untill the envionment is created:
   ```Shell
   conda env create -f environment.yml`
   ```


# Evaluation
Here are our YOLACT model pretrained on COCO dataset:

- Download Resnet101-FPN ----------[yolact_im700_54_800000.pth](https://drive.google.com/file/d/1lE4Lz5p25teiXV-6HdTiOJSnS7u7GBzg/view?usp=sharing)

I am only using Resnet101-FPN as a backbone.
To evalute the model, put the corresponding weights file in the `weights` directory and run one of the following commands.

## Results on pretrained COCO dataset
- Run the below command for evaluation
```Shell
python eval.py --trained_model=weights/yolact_im700_54_800000.pth --score_threshold=0.15 --image=car.jpeg --cross_class_nms=True --top_k=1 
```
