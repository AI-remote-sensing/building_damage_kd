# Knowldge Distillation for Building Damage Recognition 

We develop ensemble and dual-teacher knowledge distillation methods based on the "xView2: Assess Building Damage" dataset and its 1st place solution. Our paper is under reviewing.

## Environments

### basic: 
python=3.9
mongodb database(for easily saving the experiment result)

### pip package:
numpy==1.22.4
torch==1.11.0
sklearn==1.1.1
pandas==1.4.2
tqdm==4.64.0
opencv-python==4.5.5.64
torchvision==0.12.0
imgaug==0.4.0
seaborn==0.11.2
pymongo==4.1.1

### src install package:
apex from NVIDIA(Notice: pip install apex is wrong)
```
git clone https://github.com/NVIDIA/apex
cd apex
# pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
# build with C++, you may meet some dependency problem, use the following
pip install -v --no-cache-dir ./
```
Luckily, you get no error output:
> Installing collected packages: apex
> Successfully installed apex-0.1

### config setting

We use the auto emailbox for remind us the expriment process, which requrie a setting.json put in the project directory like following:

{"smtp": "smtp.163.com", "port": 465, "sender": "xxx@163.com", "passport": "YOURPASSWORD", "title": "Mingde Server", "From": "xxxx@163.com", "To": "xxx@qq.com", "Cc": [], "attachment": []}


## Dataset

You can download the compressed data file from https://www.xview2.org. You should place the uncompressed data folder "train", "tier3", "hold" and "test" into "data" folder in main working diretory.

## Data processing

This part is kept the same as the 1st place solution in xview2 challenge.

### Data Cleaning Techniques

Dataset for this competition well prepared and I have not found any problems with it.
Training masks generated using json files, "un-classified" type treated as "no-damage" (create_masks.py). "masks" folders will be created in "train" and "tier3" folders.

The problem with different nadirs and small shifts between "pre" and "post" images solved on models level:
 - Frist, localization models trained using only "pre" images to ignore this additional noise from "post" images. Simple UNet-like segmentation Encoder-Decoder Neural Network architectures used here.
 - Then, already pretrained localization models converted to classification Siamese Neural Network. So, "pre" and "post" images shared common weights from localization model and the features from the last Decoder layer concatenated to predict damage level for each pixel. This allowed Neural Network to look at "pre" and "post" separately in the same way and helped to ignore these shifts and different nadirs as well.
 - Morphological dilation with 5*5 kernel applied to classification masks. Dilated masks made predictions more "bold" - this improved accuracy on borders and also helped with shifts and nadirs.


### Data Processing Techniques

Models trained on different crops sizes from (448, 448) for heavy encoder to (736, 736) for light encoder.
Augmentations used for training:
 - Flip (often)
 - Rotation (often)
 - Scale (often)
 - Color shifts (rare)
 - Clahe / Blur / Noise (rare)
 - Saturation / Brightness / Contrast (rare)
 - ElasticTransformation (rare)

Inference goes on full image size (1024, 1024) with 4 simple test-time augmentations (original, filp left-right, flip up-down, rotation to 180).

## Training part

The folder "train_src" includes the basic training source code for teacher_building model and teacher model, knowledge distillation source code for the ST training the student_building model and student model. The checkpoint will be save into the "weights" folder in the project home directory. However, the major score data will be saved into the mongoDB for analysing and reporting.

training order:
0. create_masks
1. train_teacher_building
2. tune_teacher_building
3. train_teahcer
4. tune_teacher
5. train_student_building
6. train_student

## Inference part

The folder "inference_src" contains the code about converting the pytorch model checkpoint into ONNX format for inference, which is popular in the industry and that's why we use it for our inference time experiments. What's more, some code about get the results from database for reporting also lay here.


