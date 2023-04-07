import visualization
import nn
import precipitates
import numpy as np
import imageio
import matplotlib.pyplot as plt
import logging
import pandas as pd

import re

from tqdm.auto import tqdm
import pathlib
import  nn

import pandas as pd


import argparse

PROGRAM_NAME =  'Precipitates Distribution'
PROGRAM_DESC = 'Application desgined to recognize precipitates on images from SEM and classify their distributions' 


def _parse_args(arglist=None):

    parser = argparse.ArgumentParser(
        prog=PROGRAM_NAME,
        description=PROGRAM_DESC)

    parser.add_argument(
        '--imgfolder',
        help='Path to a folder with input images. Searched recursively.')
    parser.add_argument(
        '--modelpath',
        help='Path to a model used in precipitates identification')

    parser.add_argument(
        '--outputfolder',
        help="Path to an output folder (doesn't have to exist)")

    if arglist is not None:
        return parser.parse_args(arglist)
    else:
        return parser.parse_args()


def _get_feature_dataset(shapes):
    features = [precipitates.extract_features(shape) for shape in shapes]
    shape_classes = [ precipitates.classify_shape(feature) for feature in features]
    
    df = pd.DataFrame(features)
    df['shape_class'] = shape_classes
    return df


def _parse_px2um_scale(file_name):
    res = re.search("([0-9]+)px([0-9]+)um",str(file_name))
    groups = res.groups()
    if len(groups) !=2:
        return None
    else:
        px,um = groups
        try:
            return int(px)/int(um)
        except ValueError:
            return None
    

def _add_micrometer_scale(df_features,px2um):
    df_features['ellipse_width_um'] = df_features['ellipse_width_px'] /px2um
    df_features['ellipse_height_um'] = df_features['ellipse_height_px'] /px2um
    df_features['precipitate_area_um'] = df_features['precipitate_area_px'] /(px2um**2)

def _process_image(model, img_path,px2um,out_dir):
    
    name = img_path.stem.replace(' ','_')
    img_out_dir = out_dir/name
    img_out_dir.mkdir(exist_ok=True)
    
    img = precipitates.load_microscope_img(img_path)

    if len(img.shape) ==3:
        img = img[:,:,0]

    if img.shape != (1024,1024):
        logging.warning(f"Ignored {img_path}")
        return

    preds_mask = nn.predict(model,img)
    contoured =visualization.add_contours_morph(img,preds_mask,contour_width=2)
    
    imageio.imwrite(img_out_dir/f"{name}_contoured.png",contoured)
    imageio.imwrite(img_out_dir/f"{name}_img.png",img)
    imageio.imwrite(img_out_dir/f"{name}_mask.png",preds_mask)
    
    shapes = precipitates.identify_precipitates_from_mask(preds_mask)
    
    df_features = _get_feature_dataset(shapes)

    if px2um is not None:
        _add_micrometer_scale(df_features,px2um)
    else:
        logging.warn(f"Scale wasn't found in filename {img_path}")

    df_features.to_csv(
        img_out_dir/"precipitates.csv",
        index=False,
        header=True)

    fig_hist = visualization.plot_histograms(df_features)
    fig_hist.savefig(img_out_dir/"area_hist.pdf")
    fig_hist.clear()
    plt.close(fig_hist)
    
    fig_details = visualization.plot_precipitate_details(df_features,preds_mask,img)
    fig_details.savefig(img_out_dir/"precipitate_details.pdf")
    fig_details.clear()
    plt.close(fig_details)
    plt.close('all')


    
def _run_batch(img_paths,model,out_dir,notify_progress=False):
    scales_px2um = [_parse_px2um_scale(str(f)) for f in img_paths ]
    paths_scales = list(zip(img_paths,scales_px2um))
    
    no_scales = [str(f) for f,s in paths_scales if s is None]
    if len(no_scales) != 0:
        ls = '\n'.join(no_scales)
        logging.error(f"The following files don't have scale (e.g. \"100px20um\") in names. Add it before continuing. List:\n{ls}")
        return
    
    if notify_progress:
        paths_scales = tqdm(
            paths_scales,
            total=len(img_paths),
            desc="Processing"
        )
    
    for img_path,px2um in paths_scales:
        try:
            _process_image(model, img_path,px2um,out_dir)
        except Exception as e:
            logging.warning(f"Error while processing {img_path}. E:{e}")
  
if __name__ == "__main__":
    args = _parse_args()


    IMG_EXTS = ['.png','.jpg','.jpg','.tif','.tiff']
    img_paths = [file for file in pathlib.Path(args.imgfolder).rglob("*.*") if
        file.suffix in IMG_EXTS]
    model = nn.load_model(args.modelpath)

    out_dir = pathlib.Path(args.outputfolder)
    out_dir.mkdir(parents=True,exist_ok=True)

    _run_batch(img_paths,model,out_dir,notify_progress=True)

