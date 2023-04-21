import precipitates.img_tools as img_tools
import cv2
import numpy as np
import imageio
from dataclasses import dataclass


@dataclass
class PrecipitateFeatures:
    """Precipitate features"""
    ellipse_width_px:float
    ellipse_height_px:float
    ellipse_center_x:float
    ellipse_center_y:float
    ellipse_angle_deg:float
    circle_x:float
    circle_y:float
    circle_radius:float
    precipitate_area_px:float

@dataclass
class PrecipitateShape:
    """Precipitate mask"""
    shape_mask:np.array
    top_left_x: int
    top_left_y: int


def _crop_bottom_bar(img,bar_height = 120):
    return img[:-bar_height]

def load_microscope_img(path):
    img = np.squeeze(imageio.imread(path))
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    elif len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img,cv2.COLOR_RGBA2GRAY)
        
    width = img.shape[1]
    # ensure square
    cropped = img[:width,:]
    norm = cropped.astype(float)/np.max(cropped)*255

    return norm.astype(np.uint8)
    
def identify_precipitates_from_mask(prec_mask):
    bbs = img_tools.extract_component_with_bounding_boxes(prec_mask)
    
    shapes = []
    for (t,b,l,r),mask in bbs: 
        prec = PrecipitateShape(
            shape_mask = mask,
            top_left_x = l,
            top_left_y = t
        )
        shapes.append(prec)
    
    return shapes


def extract_features(shape:PrecipitateShape) -> PrecipitateFeatures:
    contour = img_tools.get_shape_contour(shape.shape_mask)
    (circle_x,circle_y),circle_radius = cv2.minEnclosingCircle(contour)
    
    # circle radius is from center of the circle to the CENTER of the furthest pixel
    # Since it is center of the pixel you need to add 1 to let the circle
    # reach the pixel borders on the both sides.
    circle_radius += .5
    
    if len(contour) >=5:
        (e_x,e_y),(e_width,e_height),angle = cv2.fitEllipseDirect(contour)
    else:
        e_x = circle_y
        e_y = circle_x
        e_width = circle_radius*2
        e_height = circle_radius*2
        angle = 0
    # ... >0 ensures that you count only ones, not 255 
    pixel_area = np.sum(shape.shape_mask>0)
    
    return PrecipitateFeatures(
        ellipse_width_px=e_width,
        ellipse_height_px=e_height,
        ellipse_center_x=e_x + shape.top_left_x,
        ellipse_center_y=e_y + shape.top_left_y,
        ellipse_angle_deg=angle,
        circle_x=circle_x + shape.top_left_x,
        circle_y=circle_y + shape.top_left_y,
        circle_radius=circle_radius,
        precipitate_area_px=pixel_area
    )

def classify_shape(
    features:PrecipitateFeatures,
    needle_ratio = .5,
    irregullar_threshold = .6)->str:
    
    h,w = features.ellipse_height_px,features.ellipse_width_px
    minor,major = (h,w) if h<w else (w,h)
    axis_ratio = minor/major

    if axis_ratio < needle_ratio:
        return "shape_needle"
    
    circle_area = np.pi* features.circle_radius**2
    area_ratio = features.precipitate_area_px / circle_area
    
    if area_ratio < irregullar_threshold:
        return "shape_irregular"
    else:
        return "shape_circle"

