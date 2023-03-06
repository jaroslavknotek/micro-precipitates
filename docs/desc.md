Precipitates Identification Based on Image Processing
---

# Assumption

- Input is a grayscale image with precipitates being black. 
- Precipitates have area larger than 3x3 px

# Method

Given the input image, all dark spots smaller than 3x3 are removed by applying [morphological operator close](https://www.ni.com/docs/en-US/bundle/ni-vision-concepts-help/page/grayscale_morphology.html). Next, the precipitates are extracted using thresholding with arbitrary value. This operation results in a mask which is an image, of the same size as the input, with white color where precipitates are present and zeros where not precipitate was found. In general, each precipitate is represented as an "island" of white in a "sea" of zeros and can be easily extracted. 

For each individual precipitate shape, the following three properties are calculated:
- circumscribed circle
- approximating ellipse
- area as a sum of all pixels

Such features are used for a classification of shape type. A precipitate has shape of:
- circle - if ellipse's axes are of similar length and area of precipitate is similar to the one of the circumscribed circle
- needle - ellipse width is twice its height or vice versa
- irregular - otherwise


# Application Output Description

For each image, there exist a folder with the same name in `/outputs`. Each folder contains the following files:

- Input image `img.png`
- Extracted data `data.csv`
- Visualized data `highlight.png`
- Detailed view of precipitates `details.pdf`
- Precipitate mask `mask.png`
- Chart with distributions `dist.png`

## Input Image

An image from SEM conforming to the assumption mentioned above.

## Extracted Data

A CSV file that contains record describing visual properties of precipitates. Each record has the following properties:

- `shape_class` - A shape of the precipitate (one of `shape_irregular`/`shape_circle`/`shape_needle`).
- `circle_x`, `circle_y`, `circle_radius` - Center and radius of a circle circumscribed around the precipitate
- `precipitate_area_px` - Area of the precipitate measured in pixels
- `precipitate_area_ratio` - Relative area of the precipitate. Calculated as `precipitate_area_px` divided by number of image pixels.
- `ellipse_width_px`,`ellipse_height_px` - Lengths of the ellipse's axes.
- `ellipse_center_x`,`ellipse_center_y` - Center of the ellipse.
- `ellipse_angle_deg` - Ellipse rotation

## Visualized Data

An image `highlight.png` with precipitates highlighted by ellipses. Each ellipse has color based on the shape of the precipitate:
- Red - `Needle`
- Green - `Circular`
- Magenta - `Irregular`

The individual precipitates are cropped out and visualized on `detailed.pdf`. Note, the technical issues prevent us from showing more than 200 precipitates.

## Distributions

The file `dist.png` contains two charts
- Area distribution - histogram showing relative areas (`precipitate_area_ratio`) of each precipitate
- Shapes distribution - bar chart with number of precipitates of each `shape_class`

## Precipitate Mask

Binary image that shows white where a precipitate was identified (base on the method explained above) and black where no precipitate was found.

