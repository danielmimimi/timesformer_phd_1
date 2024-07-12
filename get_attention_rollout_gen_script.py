import argparse
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
from visualize_attn_util_2 import DividedAttentionRollout, create_masks, get_frames
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, accuracy_score
from PIL import Image, ImageDraw


def draw_grid(image, grid_size,width=None,height=None):
    draw = ImageDraw.Draw(image)
    
    if width == None and height == None:
        width, height = image.size
        cell_width = width // grid_size
        cell_height = height // grid_size

        for i in range(1, grid_size):
            # Vertical lines
            draw.line([(i * cell_width, 0), (i * cell_width, height)], fill="blue", width=1)
            draw.line([(0, i * cell_height), (width, i * cell_height)], fill="blue", width=1)


    else:
        cell_width = width // (grid_size*8)
        cell_height = height // grid_size

        for i in range(1, grid_size*8):
            # Vertical lines
            draw.line([(i * cell_width, 0), (i * cell_width, height)], fill="blue", width=1)
        for i in range(1, grid_size):
            # Horizontal lines
            draw.line([(0, i * cell_height), (width, i * cell_height)], fill="blue", width=1)

    return image

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


def transform_plot(image_path):
    # Read the image
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    
    # Convert the image to a tensor
    tensor = transforms.ToTensor()(image)
    
    # Interpolate (resize) the tensor
    resized_tensor = F.interpolate(tensor.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
    
    # Convert the tensor back to numpy array
    final_image = rearrange(resized_tensor * 255, 'c h w -> h w c').numpy()
    
    return final_image


def read_args():

    parser = argparse.ArgumentParser(
                        prog='ProgramName',
                        description='What the program does',
                        epilog='Text at the bottom of help')
    parser.add_argument('-cp', '--checkpoint_file')      # option that takes a value
    parser.add_argument('-t', '--model_description')  
    parser.add_argument('-c', '--configs')  
    parser.add_argument('-m', '--min_amount_of_data',default=20,type=int)  
    args = parser.parse_args()
    return args

def main(args):
    model_files = [args.checkpoint_file]
    # Data Model Sampling
    # "REAL_ORIGINAL_UNIFORM",
    texts = [args.model_description]

    configs = [args.configs]


    output_dir = Path('/workspaces/TimeSFormer/final')
    output_dir.mkdir(parents=True, exist_ok=True)

    for config,model_file,model_name  in zip(configs,model_files,texts): 

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


        model_name =  model_name+"_{}".format(model_name)
        model_output_dir = output_dir / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)

        html_content = "<html><head><title>Model Evaluation from {}</title></head><body>".format(model_name)
        html_content += "<h2>Overview of Labels</h2><ul>"

        html_content += "<h2>Table of Contents</h2><ul>"
        html_content += '<li><a href="#overview">Overview of Labels</a></li>'
        for frame_index in tqdm(range(0,3)):
            html_content += f'<li><a href="#frame_{frame_index}">Frame {frame_index}</a></li>'
            html_content += f'<li><a href="#positive_frame_{frame_index}">Positive Classified {frame_index}</a></li>'
            html_content += f'<li><a href="#negative_frame_{frame_index}">Negative Classified {frame_index}</a></li>'
        html_content += "</ul>"


        html_content += '<h2 id="overview">Overview of Labels</h2><ul>'
        with PathManager.open(annotation_file, "r") as f:
            label_mapping = json.load(f)
            for label, description in label_mapping.items():
                html_content += f"<li>{label}: {description}</li>"
            html_content += "</ul>"

        # Accuracy of the model 

        cfg.TRAIN.ENABLE = False
        cfg.TIMESFORMER.PRETRAINED_MODEL = model_file
        cfg.TRAIN.CHECKPOINT_FILE_PATH = model_file
        model = MODEL_REGISTRY.get('vit_base_patch16_224')(cfg)

        device = torch.device('cuda:0')
        model.to(device)

        with torch.set_grad_enabled(False):
            np.random.seed(cfg.RNG_SEED)
            torch.manual_seed(cfg.RNG_SEED)
            model.eval()

        with PathManager.open(label_file, "r") as f:
            label_json = json.load(f)





        att_roll = DividedAttentionRollout(model,device)
        for frame_index in tqdm(range(0,40 if len(hslu_dataset) > 40 else len(hslu_dataset))):
            start,stop = hslu_dataset.__get_frame_info__(frame_index)
            y_true = []
            y_pred = []
            fp = []
            tp = []
            fn = []

            

            if cfg['DATA']['SAMPLING_METHOD'] == 'rolling':
                for start_index in range(start,start+5):
                    frame, label,index,_ = hslu_dataset.__getitem_rolling__(frame_index,start_index)
                    masks,prediction,(attentions_space_mean,attentions_temporal_mean) = att_roll(frame)

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

                    # SPATIAL
                    spatial_combined = np.hstack(np_imgs)
                    spatial_mask = create_masks(attentions_space_mean,np_imgs)
                    spatial_combined = np.vstack((spatial_combined,  np.hstack(spatial_mask))).astype(np.uint8)

                    # TEMPORAL 
                    temporal_combined = np.hstack(np_imgs)
                    temporal_mask = create_masks(list(rearrange(attentions_temporal_mean, 'h w t -> t h w')),np_imgs)
                    temporal_combined = np.vstack((temporal_combined,  np.hstack(temporal_mask))).astype(np.uint8)
                    

                    first_image = Image.fromarray(combined)


                    width, height = first_image.size
                    grid_image = draw_grid(first_image, 14,width=width,height=224)
                    combined = np.array(grid_image)

                    # Temporal 
                    max_index_image = np.argmax(attentions_temporal_mean, axis=2)
                    std_dev_image = np.std(attentions_temporal_mean, axis=2)
                    std_temporal_resized = cv2.resize(std_dev_image, (224, 224))
                    std_temporal_resized_i = np.uint8(255 * std_temporal_resized / np.max(std_temporal_resized))
                    temporal_heatmap_std = cv2.applyColorMap(std_temporal_resized_i, cv2.COLORMAP_JET)
                    temporal_heatmap_std = np.float32(temporal_heatmap_std)

                    stacked_imgs = [np.expand_dims(p,0) for p in np_imgs]
                    stacked_imgs = np.vstack(stacked_imgs)
                    mean_image = np.uint8(np.mean(stacked_imgs, axis=0))
                    differences = np.abs(stacked_imgs - mean_image)
                    difference_image = np.mean(differences, axis=0)
                    difference_image_uint8 = np.uint8(difference_image)

                    mean_image = temporal_heatmap_std + mean_image
                    mean_image = mean_image / np.max(mean_image)
                    mean_image_temporal_std_overlay = np.uint8(255*mean_image)

                    

                    combined_image_path =  model_output_dir  / f"combined_{model_name}_{folder_name}_{start_index}.png"
                    combined = combined[:,:,::-1]
                    Image.fromarray(combined).save(combined_image_path)
                    relative_combined_image_path = combined_image_path.relative_to(output_dir)

                    combined_spatial_image_path =  model_output_dir  / f"combined_spatial_{model_name}_{folder_name}_{start_index}.png"
                    spatial_combined = spatial_combined[:,:,::-1]
                    Image.fromarray(spatial_combined).save(combined_spatial_image_path)
                    relative_combined_spatial_image_path = combined_spatial_image_path.relative_to(output_dir)


                    combined_temporal_image_path =  model_output_dir  / f"combined_temporal_{model_name}_{folder_name}_{start_index}.png"
                    temporal_combined = temporal_combined[:,:,::-1]
                    Image.fromarray(temporal_combined).save(combined_temporal_image_path)
                    relative_combined_temporal_image_path = combined_temporal_image_path.relative_to(output_dir)

                    temporal_heatmap_image_path =  model_output_dir  / f"combined_temporal_hm_{model_name}_{folder_name}_{start_index}.png"
                    mean_image_temporal_std_overlay = mean_image_temporal_std_overlay[:,:,::-1]
                    Image.fromarray(mean_image_temporal_std_overlay).save(temporal_heatmap_image_path)
                    relative_combined_temporal_hm_image_path = temporal_heatmap_image_path.relative_to(output_dir)


                    if class_of_prediction == label:
                        tp.append({'label':label,'prediction':class_of_prediction,"image":relative_combined_image_path,'temporal_heatmap':relative_combined_temporal_hm_image_path,'spatial_attention':relative_combined_spatial_image_path,'temporal_attention':relative_combined_temporal_image_path,'folder_name':folder_name,'start_frame':start_index})
                    else:
                        fn.append({'label':label,'prediction':class_of_prediction,"image":relative_combined_image_path,'temporal_heatmap':relative_combined_temporal_hm_image_path,'spatial_attention':relative_combined_spatial_image_path,'temporal_attention':relative_combined_temporal_image_path,'folder_name':folder_name,'start_frame':start_index})
            else:
                frame, label,index,_ = hslu_dataset.__getitem__(frame_index)
                masks,prediction,(attentions_space_mean,attentions_temporal_mean) = att_roll(frame)
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

                # SPATIAL
                spatial_combined = np.hstack(np_imgs)
                spatial_mask = create_masks(attentions_space_mean,np_imgs)
                spatial_combined = np.vstack((spatial_combined,  np.hstack(spatial_mask))).astype(np.uint8)

                # TEMPORAL 
                temporal_combined = np.hstack(np_imgs)
                temporal_mask = create_masks(list(rearrange(attentions_temporal_mean, 'h w t -> t h w')),np_imgs)
                temporal_combined = np.vstack((temporal_combined,  np.hstack(temporal_mask))).astype(np.uint8)
                

                first_image = Image.fromarray(combined)


                width, height = first_image.size
                grid_image = draw_grid(first_image, 14,width=width,height=224)
                combined = np.array(grid_image)

                # Temporal 
                max_index_image = np.argmax(attentions_temporal_mean, axis=2)
                std_dev_image = np.std(attentions_temporal_mean, axis=2)
                std_temporal_resized = cv2.resize(std_dev_image, (224, 224))
                std_temporal_resized_i = np.uint8(255 * std_temporal_resized / np.max(std_temporal_resized))
                temporal_heatmap_std = cv2.applyColorMap(std_temporal_resized_i, cv2.COLORMAP_JET)
                temporal_heatmap_std = np.float32(temporal_heatmap_std)

                # stacked_imgs = [np.expand_dims(p,0) for p in np_imgs]
                # stacked_imgs = np.vstack(stacked_imgs)
                # mean_image = np.uint8(np.mean(stacked_imgs, axis=0))

                stacked_imgs = [np.expand_dims(p,0) for p in np_imgs]
                stacked_imgs = np.vstack(stacked_imgs)
                mean_image = np.uint8(np.mean(stacked_imgs, axis=0))

                mean_image = temporal_heatmap_std + mean_image
                mean_image = mean_image / np.max(mean_image)
                mean_image_temporal_std_overlay = np.uint8(255*mean_image)

                # Pairwise Change detection
                stacked_imgs_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in np_imgs]
                pairwise_diffs = [np_imgs[0]]
                for i in range(len(stacked_imgs_gray) - 1):
                    diff = np.abs(stacked_imgs_gray[i] - stacked_imgs_gray[i + 1])
                    pairwise_diffs.append(cv2.applyColorMap(((diff/np.max(diff))*255).astype(np.uint8), cv2.COLORMAP_JET))#cv2.cvtColor((diff*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                asdas = pairwise_diffs[0]
                temporal_combined = np.vstack((temporal_combined,  np.hstack(pairwise_diffs))).astype(np.uint8)

                start_index = 0 # Here just one

                combined_image_path =  model_output_dir  / f"combined_{model_name}_{folder_name}_{start_index}.png"
                combined = combined[:,:,::-1]
                Image.fromarray(combined).save(combined_image_path)
                relative_combined_image_path = combined_image_path.relative_to(output_dir)

                combined_spatial_image_path =  model_output_dir  / f"combined_spatial_{model_name}_{folder_name}_{start_index}.png"
                spatial_combined = spatial_combined[:,:,::-1]
                Image.fromarray(spatial_combined).save(combined_spatial_image_path)
                relative_combined_spatial_image_path = combined_spatial_image_path.relative_to(output_dir)


                combined_temporal_image_path =  model_output_dir  / f"combined_temporal_{model_name}_{folder_name}_{start_index}.png"
                temporal_combined = temporal_combined[:,:,::-1]
                Image.fromarray(temporal_combined).save(combined_temporal_image_path)
                relative_combined_temporal_image_path = combined_temporal_image_path.relative_to(output_dir)

                temporal_heatmap_image_path =  model_output_dir  / f"combined_temporal_hm_{model_name}_{folder_name}_{start_index}.png"
                mean_image_temporal_std_overlay = mean_image_temporal_std_overlay[:,:,::-1]
                Image.fromarray(mean_image_temporal_std_overlay).save(temporal_heatmap_image_path)
                relative_combined_temporal_hm_image_path = temporal_heatmap_image_path.relative_to(output_dir)


                if class_of_prediction == label:
                    tp.append({'label':label,'prediction':class_of_prediction,"image":relative_combined_image_path,'temporal_heatmap':relative_combined_temporal_hm_image_path,'spatial_attention':relative_combined_spatial_image_path,'temporal_attention':relative_combined_temporal_image_path,'folder_name':folder_name,'start_frame':start_index})
                else:
                    fn.append({'label':label,'prediction':class_of_prediction,"image":relative_combined_image_path,'temporal_heatmap':relative_combined_temporal_hm_image_path,'spatial_attention':relative_combined_spatial_image_path,'temporal_attention':relative_combined_temporal_image_path,'folder_name':folder_name,'start_frame':start_index})

            # Create and save confusion matrix
            
            html_content += f'<h2 id="frame_{frame_index}">Frame {frame_index}</h2>'
            cm = confusion_matrix(y_true, y_pred)
            cm_image_path = model_output_dir  / f"confusion_matrix_{model_name}_{folder_name}.png"
            plot_confusion_matrix(cm, classes=np.unique(y_true), model_name=model_name, save_path=cm_image_path)
            relative_cm_image_path = cm_image_path.relative_to(output_dir)
            html_content += f'<h2>Confusion Matrix for {model_name}</h2>'
            html_content += f'<img src="{relative_cm_image_path}" alt="Confusion Matrix for {model_name}"><br>'
            html_content += f'<h2 id="negative_frame_{frame_index}"">Negative Sample</h2>'
            for false_negative_sample in fn:
                relative_combined_image_path = false_negative_sample["image"]
                class_of_prediction = false_negative_sample["prediction"]
                folder_name =  false_negative_sample["folder_name"]
                start_index = str(false_negative_sample["start_frame"])
                html_content += f"<h3>{model_name} - {start_index} - Folder: {folder_name} :::::: Predicted {label_mapping[str(class_of_prediction)]}</h3>"

                spatial_attention_path = false_negative_sample["spatial_attention"]
                temporal_attention_path = false_negative_sample["temporal_attention"]
                temporal_heatmap_path = false_negative_sample["temporal_heatmap"]

                html_content += '<p> The following images are showing the joint space time attention rollout </p>'
                html_content += '<ul><li>Original Image with drawn Grid (14x14)</li><li>Mean Attention Rollout</li><li>Min Attention Rollout</li><li>Max Attention Rollout</li> </ul>'
                                
                                
                                
                            
                html_content += f'<img src="{relative_combined_image_path}" alt="{model_name} - {folder_name}"><br>'

                html_content += '<p> The following images are showing the spatial attention rollout </p>'
                html_content += '<ul><li>Original Image</li><li>Mean Attention Rollout</li> </ul>'
                                
                html_content += f'<img src="{spatial_attention_path}" alt="{model_name} - {folder_name}"><br>'

                html_content += '<p> The following images are showing the temporal attention rollout </p>'
                html_content += '<ul><li>Standard deviation Over Temporal Attention Rollout (14x14x8)</li><li>Original Image</li><li>Mean Attention Rollout</li><li>Pairwise Change Detection</li> </ul>'
                html_content += f'<img src="{temporal_heatmap_path}" alt="{model_name} - {folder_name}"><br>'
                html_content += f'<img src="{temporal_attention_path}" alt="{model_name} - {folder_name}"><br>'

            html_content += f'<h2 id="positive_frame_{frame_index}">Positive Sample</h2>'
            for true_positive_sample in tp:
                relative_combined_image_path = true_positive_sample["image"]
                class_of_prediction = true_positive_sample["prediction"]
                folder_name =  true_positive_sample["folder_name"]
                start_index = str(true_positive_sample["start_frame"])
                html_content += f"<h3>{model_name} - {start_index} - Folder: {folder_name} :::::: Predicted {label_mapping[str(class_of_prediction)]}</h3>"
                
                html_content += '<p> The following images are showing the joint space time attention rollout </p>'
                html_content += '<ul><li>Original Image with drawn Grid (14x14)</li><li>Mean Attention Rollout</li><li>Min Attention Rollout</li><li>Max Attention Rollout</li> </ul>'
                                
                html_content += f'<img src="{relative_combined_image_path}" alt="{model_name} - {folder_name}"><br>'

                spatial_attention_path = true_positive_sample["spatial_attention"]
                temporal_attention_path = true_positive_sample["temporal_attention"]
                temporal_heatmap_path = true_positive_sample["temporal_heatmap"]


                html_content += '<p> The following images are showing the spatial attention rollout </p>'
                html_content += '<ul><li>Original Image</li><li>Mean Attention Rollout</li> </ul>'
                html_content += f'<img src="{spatial_attention_path}" alt="{model_name} - {folder_name}"><br>'

                html_content += '<p> The following images are showing the temporal attention rollout </p>'
                html_content += '<ul><li>Standard deviation Over Temporal Attention Rollout (14x14x8)</li><li>Original Image</li><li>Mean Attention Rollout</li> </ul>'
                
                html_content += f'<img src="{temporal_heatmap_path}" alt="{model_name} - {folder_name}"><br>'
                html_content += f'<img src="{temporal_heatmap_path}" alt="{model_name} - {folder_name}"><br>'



        html_content += "</body></html>"
        html_file_path =output_dir / "model_evaluation_{}.html".format(model_name)
        with open(html_file_path, "w") as html_file:
            html_file.write(html_content)
        print(f"HTML file generated successfully: {html_file_path}")



if __name__ == "__main__":
    args = read_args()
    main(args)
