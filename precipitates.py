import img_tools
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

def _improve_prec_mask_detail(
    precipitate_mask,
    precipitate_image,
    shape_threshold):
    
    shape_raw = precipitate_image<=shape_threshold
    
    shape_fill_gaps = np.bitwise_or(shape_raw,precipitate_mask)
    
    dilate_mask = cv2.dilate(precipitate_mask ,np.ones((3,3)))
    return np.bitwise_and(shape_fill_gaps,dilate_mask).astype(np.uint8)


def load_microscope_img(path,has_bottom_bar=True):
    img = imageio.imread(path)
    if has_bottom_bar:
        img = _crop_bottom_bar(img)
    return img_tools.norm(img)

def extract_raw_mask(img,threshold,min_prec_size = 3):
    bcg_normed = img_tools.background_divide(img)
    gs_closed = img_tools.morph_grayscale_close(bcg_normed, min_prec_size)
    img_normed = img_tools.norm(gs_closed)
    return img_tools.threshold(img_normed, threshold)

def identify_precipitates(img,threshold):
    
    # this mask is low res because of morphologocy operation (CLOSE)
    # used in extract raw mask
    prec_mask_low_res = extract_raw_mask(img,threshold)
    bbs = img_tools.extract_component_with_bounding_boxes(prec_mask_low_res)
    
    prec_mask = np.zeros(img.shape)
    shapes = []
    for (t,b,l,r),mask in bbs:
        precipitate_image = img[t:b,l:r]
    
        detail_prec_mask = _improve_prec_mask_detail(
            mask,
            precipitate_image,
            threshold
        )
    
        prec_mask[t:b,l:r] = detail_prec_mask
        prec = PrecipitateShape(
            shape_mask = detail_prec_mask,
            top_left_x = l,
            top_left_y = t
        )
        shapes.append(prec)
    
    return prec_mask, shapes


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
    
    pixel_area = np.sum(shape.shape_mask)
    
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
