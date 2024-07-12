from io import BytesIO
import json
import os
from pathlib import Path

import cv2
from fvcore.common.file_io import PathManager
from matplotlib import pyplot as plt
from tqdm import tqdm
from timesformer.datasets.hslu import Hslu
from timesformer.models.vit import *
from timesformer.config.defaults import get_cfg
from visualize_attn_util import DividedAttentionRollout, create_masks, get_frames
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, accuracy_score
from PIL import Image

def plot_confusion_matrix(cm, classes, model_name, save_path):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion matrix for {model_name}')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_path)
    plt.close()

transform_plot = transforms.Compose([
    lambda p: cv2.imread(str(p),cv2.IMREAD_COLOR),
    transforms.ToTensor(),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    lambda x: rearrange(x*255, 'c h w -> h w c').numpy()
])

model_files = [
'/workspaces/TimeSFormer/checkpoints/checkpoint_epoch_00040_gen2_half_step.pyth',
'/workspaces/TimeSFormer/checkpoints/checkpoint_epoch_00050_gen2_one_step.pyth',
'/workspaces/TimeSFormer/checkpoints/checkpoint_epoch_00020_gen2_uniform.pyth'
]

texts = ["Finetuned Half Step","Finetuned One Step","Finetuned Uniform"]
frame_distances = ["half_step","one_step","uniform"]

video_path = Path('/workspaces/TimeSFormer/timesformer/datasets/hslu/gen2')

configs = [
    '/workspaces/TimeSFormer/configs/hslu/TimeSformer_divST_8_224_gen2_half_step.yaml',
    '/workspaces/TimeSFormer/configs/hslu/TimeSformer_divST_8_224_gen2_one_step.yaml',
    '/workspaces/TimeSFormer/configs/hslu/TimeSformer_divST_8_224_gen2.yaml']





output_dir = Path('/workspaces/TimeSFormer/model_evaluation_output_gen2')
output_dir.mkdir(parents=True, exist_ok=True)

for config,model_file,model_name,frame_distance  in zip(configs,model_files,texts,frame_distances): 

    cfg = get_cfg()
    cfg.merge_from_file(config)
    label_file = os.path.join(
                cfg.DATA.PATH_TO_DATA_DIR,
                "annotations_test_{}.json".format(
                    cfg.DATA.SAMPLING_METHOD
                ),
            )


    annotation_file = os.path.join(
        cfg.DATA.PATH_TO_DATA_DIR,
        "label_files.json",
    )

    hslu_dataset = Hslu(cfg,mode="test")


    model_name =  model_name+"_Distance_{}".format(frame_distance)
    model_output_dir = output_dir / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    html_content = "<html><head><title>Model Evaluation from {}</title></head><body>".format(model_name)
    html_content += "<h2>Overview of Labels</h2><ul>"

    with PathManager.open(annotation_file, "r") as f:
        label_mapping = json.load(f)
        for label, description in label_mapping.items():
            html_content += f"<li>{label}: {description}</li>"
        html_content += "</ul>"

    # Confusions Matrix
    cm_image_path = model_output_dir  / f"confusion_matrix_{model_name}.png"
    relative_cm_image_path = cm_image_path.relative_to(output_dir)
    html_content += f"<h2>Confusion Matrix for {model_name}</h2>"
    html_content += f'<img src="{relative_cm_image_path}" alt="Confusion Matrix for {model_name}"><br>'

    # Accuracy of the model 

    cfg.TRAIN.ENABLE = False
    cfg.TIMESFORMER.PRETRAINED_MODEL = model_file
    model = MODEL_REGISTRY.get('vit_base_patch16_224')(cfg)

    device = torch.device('cuda:0')
    model.to(device)

    with torch.set_grad_enabled(False):
        np.random.seed(cfg.RNG_SEED)
        torch.manual_seed(cfg.RNG_SEED)
        model.eval()

    with PathManager.open(label_file, "r") as f:
        label_json = json.load(f)


    y_true = []
    y_pred = []

    

    att_roll = DividedAttentionRollout(model,device)
    for frame_index in tqdm(range(0,len(hslu_dataset))):
        frame, label,index,_ = hslu_dataset.__getitem__(frame_index)
        masks,prediction = att_roll(frame)

        class_of_prediction = prediction.argmax().item()
     
        folder_name = Path(_['frame_paths'][0]).parent.name

        class_of_prediction = prediction.argmax().item()

        y_true.append(label)
        y_pred.append(class_of_prediction)

        np_imgs = [transform_plot(p) for p in _['frame_paths']]
        masks_stacked = [] # mean,min,max
        combined = np.hstack(np_imgs)
        for mask_index in range(len(masks)):
        # np_imgs = [np_imgss[i].numpy() for i in range(np_imgss.shape[0])]
            mask_image = create_masks(list(rearrange(masks[mask_index], 'h w t -> t h w')),np_imgs)
            stacked_masks = np.hstack(mask_image)
            masks_stacked.append(stacked_masks)

            combined = np.vstack((combined, stacked_masks)).astype(np.uint8)

        
        
        # Stack the combined images and masks vertically
       
        # combined = np.vstack((stacked_imgs, stacked_masks)).astype(np.uint8)


        combined_image_path =  model_output_dir  / f"combined_{model_name}_{folder_name}.png"
        combined = combined[:,:,::-1]
        Image.fromarray(combined).save(combined_image_path)

        relative_combined_image_path = combined_image_path.relative_to(output_dir)
        html_content += f"<h3>{model_name} - Folder: {folder_name} :::::: Predicted {label_mapping[str(class_of_prediction)]}</h3>"
        html_content += f'<img src="{relative_combined_image_path}" alt="{model_name} - {folder_name}"><br>'
  
  
    # Create and save confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, classes=np.unique(y_true), model_name=model_name, save_path=cm_image_path)

    # Save Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    html_content += f"<h3>Accuracy for {model_name}: {accuracy:.2f}</h3>"

    html_content += "</body></html>"
    html_file_path =output_dir / "model_evaluation_{}.html".format(model_name)
    with open(html_file_path, "w") as html_file:
        html_file.write(html_content)
    print(f"HTML file generated successfully: {html_file_path}")

# for  model_file,img_name in zip(model_files,texts): 
#   # model_file = '/workspaces/TimeSFormer/Startpoints/TimeSformer_divST_8x32_224_K400.pyth'
#   assert Path(model_file).exists()


#   cfg.TRAIN.ENABLE = False
#   cfg.TIMESFORMER.PRETRAINED_MODEL = model_file
#   model = MODEL_REGISTRY.get('vit_base_patch16_224')(cfg)

#   with torch.set_grad_enabled(False):
#     np.random.seed(cfg.RNG_SEED)
#     torch.manual_seed(cfg.RNG_SEED)
#     model.eval()
#   #   pred = model(create_video_input(path_to_video)).cpu().detach()

#   path_to_video = Path(video_path)
#   path_to_video.exists()
#   att_roll = DividedAttentionRollout(model)
#   masks = att_roll(path_to_video)


#   np_imgs = [transform_plot(p) for p in get_frames(path_to_video)]
#   masks = create_masks(list(rearrange(masks, 'h w t -> t h w')),np_imgs)

#   stacked_imgs = np.hstack(np_imgs)
#   stacked_masks = np.hstack(masks)
#   # Stack the combined images and masks vertically
#   combined = np.vstack((stacked_imgs, stacked_masks)).astype(np.uint8)

#   cv2.imwrite(img_name+".png",combined)