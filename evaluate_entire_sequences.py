import argparse
import json
import os
from pathlib import Path
import cv2

from fvcore.common.file_io import PathManager
from annotation_handler_hslu import AnnotationHandlerHslu
from timesformer.config.defaults import get_cfg
from timesformer.datasets import utils
from timesformer.models.vit import *
import pandas as pd
import matplotlib.pyplot as plt

def read_args():

    parser = argparse.ArgumentParser(
                        prog='ProgramName',
                        description='What the program does',
                        epilog='Text at the bottom of help')
    parser.add_argument('-cp', '--checkpoint_files', nargs='+')      # option that takes a value
    parser.add_argument('-t', '--model_descriptions', nargs='+')  
    parser.add_argument('-c', '--standard_config') 
    # PATH_TO_DATA_DIR, PATH_PREFIX
    # SAMPLING_METHOD
    parser.add_argument('-s', '--sampling_method') 
    parser.add_argument('-p', '--path_to_dataset') 
    parser.add_argument('-d', '--dataset',default="hslu") 

    parser.add_argument('-m', '--min_amount_of_data',default=20,type=int)  

    parser.add_argument('-o', '--output_path')
    parser.add_argument('--dataset_class',default="hslu") 
    args = parser.parse_args()
    return args

def get_model_config(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.standard_config)

    cfg.DATA.SAMPLING_METHOD = args.sampling_method
    cfg.DATA.PATH_TO_DATA_DIR = args.path_to_dataset
    cfg.DATA.PATH_PREFIX = args.path_to_dataset
    cfg.TEST.DATASET = args.dataset
    cfg.TRAIN.DATASET = args.dataset


    # Accuracy of the model 
    cfg.TRAIN.ENABLE = False
    return cfg

def load_model(cfg,model_file):

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
    return model,device

def get_center_point_of_annotations(annotation_items,image_names):
    points = []
    for annotation_item in annotation_items:
         if type(annotation_item) == AnnotationHandlerHslu.annotationItemPoint and 'walking_root'in annotation_item.label and not 'world' in  annotation_item.label :
            if (annotation_item.imageName.split(":")[0] in image_names):
                points.append(annotation_item.point)
    data_array = np.array(points)
    mean_values = np.mean(data_array, axis=0)
    return mean_values.astype(int)



def main(args):
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = get_model_config(args)
    model,device = load_model(cfg,args.checkpoint_files[0])


    dataset = os.listdir(args.path_to_dataset)

    annotation_file = os.path.join(
            args.path_to_dataset,
            "label_files.json",
        )
    
    with PathManager.open(annotation_file, "r") as f:
        label_mapping = json.load(f)
        
    results = []
    for data in dataset:
        print(data)
        if ".json" in data:
            continue
        dataset_path = Path(args.path_to_dataset).joinpath(data).as_posix()
        annotations_file = [file for file in os.listdir(dataset_path) if file.endswith('.csv')]
        annotation_handler = AnnotationHandlerHslu()
        annotation_path = Path(args.path_to_dataset).joinpath(data).joinpath(annotations_file[0])
        annotation_handler.readAnnotations(annotation_path.as_posix())
        annotation_items = annotation_handler.getAnnotationItemsList()
        png_files = [file for file in os.listdir(dataset_path) if file.endswith('.png')]
        batch_size = 8
        for start in range(len(png_files) - batch_size + 1):
            batch = png_files[start:start+batch_size]

            center_point_of_annotations = get_center_point_of_annotations(annotation_items,batch)

            dummy_image_to_get_size = cv2.imread(Path(args.path_to_dataset).joinpath(data).joinpath(batch[0]).as_posix())
            image_size = dummy_image_to_get_size.shape
            frames = torch.as_tensor(
                utils.retry_load_images(
                    [Path(args.path_to_dataset).joinpath(data).joinpath(frame) for frame in batch],
                    3,
                )
            )
            frames = utils.tensor_normalize(
                frames, cfg.DATA.MEAN, cfg.DATA.STD
            )
            frames = frames.permute(3, 0, 1, 2)
            crop_size = 224
            frames = utils.simple_scale(frames,crop_size)
            frames = torch.index_select(
                 frames,
                 1,
                 torch.linspace(
                     0, frames.shape[1] - 1, cfg.DATA.NUM_FRAMES

                 ).long(),)
            frames = frames.unsqueeze(dim=0)
            model.zero_grad()
            prediction = model(frames.to(device))
            class_of_prediction = prediction.argmax().item()
            print(label_mapping[str(class_of_prediction)])
            results.append({'centerpoint_x':center_point_of_annotations[0],'centerpoint_y':center_point_of_annotations[1],"dataset":data,"prediction":label_mapping[str(class_of_prediction)],'image_size':image_size})
    result_dataframe = pd.DataFrame(results)
    datasets = result_dataframe['dataset'].unique()
    colors = plt.cm.get_cmap('tab20c', len(datasets))  # Choose a colormap with enough colors
    plt.figure(figsize=(10, 7))
    for i, dataset in enumerate(datasets):
        df_subset = result_dataframe[result_dataframe['dataset'] == dataset]
        
        # Extract relevant data
        centerpoint_x = df_subset['centerpoint_x'].values
        centerpoint_y = df_subset['centerpoint_y'].values
        prediction = df_subset['prediction'].values
    
        plt.plot(centerpoint_x, centerpoint_y, marker='o', color=colors(i), label=dataset)
            
        for x, y, pred in zip(centerpoint_x, centerpoint_y, prediction):
            if pred != 'none':
                plt.plot(x, y, marker='o', color='black', markersize=8)  # Mark non-'none' predictions as black
                # plt.text(x, y, pred, fontsize=9, verticalalignment='bottom', horizontalalignment='right', color='black')
            # else:
            #     plt.text(x, y, pred, fontsize=9, verticalalignment='bottom', horizontalalignment='right')

    # Set labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Center Points with Predictions (Black - Open)')

    plt.xlim(0, 640)
    plt.ylim(0, 480)
    plt.gca().invert_yaxis()



    # Show grid
    plt.grid(True)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Legend outside, adjust the anchor position as needed


    plt.savefig("fig.png", bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    args = read_args()
    main(args)
