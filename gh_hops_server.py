from flask import Flask
import ghhops_server as hs
import rhino3dm
import cv2 as cv
import numpy as np

# register hops app as middleware
app = Flask(__name__)
hops: hs.HopsFlask = hs.Hops(app)


# flask app can be used for other stuff drectly
@app.route("/help")
def help():
    return "Welcome to Grashopper Hops for CPython!"


@hops.component(
    "/opencv",
    name="OpenCV",
    nickname="OpenCV",
    description="OpenCV Component with CPython",
    icon="/Users/yasinturedi/Desktop/Grasshopper_Open_CV_Component/open_cv.png",
    inputs=[
        hs.HopsString("a", "Main Image", "GH HOPS SOURCE OF THE IMAGE"),
        hs.HopsInteger("b", "K", "K-MEANS CLUSTERING NUMBER"),
        hs.HopsString("c", "Created Image Directory", "CREATED IMAGES DIR"),
        hs.HopsString("d", "Created Image Name", "CREATED IMAGE NAME"),
        hs.HopsString("e", "Image Format", "CREATED IMAGE FORMAT"),
        hs.HopsInteger("f", "Treshold Number", "NUMBER OF TRESHOLD"),
        hs.HopsInteger("g", "Contour Index", "INDEX OF CONTOUR"),
        hs.HopsInteger("h", "Contour Stroke", "THICKNESS OF CONTOUR"),
        hs.HopsString("j", "Contoured Image Name", "NAME OF THE CONTOURED IMAGE "),
        hs.HopsString("k", "Sketched Image Name", "NAME OF THE SKETCHED IMAGE")
    ],
    outputs=[
        hs.HopsString("l", "Image K Means", "Image of K-Means"),
        hs.HopsString("m", "Image Contour", "Image of Contour"),
        hs.HopsString("n", "Image Sketch", "Image of Sketch")
    ]
)
def myiamge(main_img,K,c_image_dir,c_image_name,img_format,treshold_number,contour_index,contour_stroke,c_image_contour_name,c_image_sketch_name): 
    img = cv.imread(main_img)
    # convert to np.float32
    Z = np.float32(img.reshape((-1,3)))
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    cv.imwrite(c_image_dir + c_image_name + img_format, res2)
    ####
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    black_image = np.zeros((img_gray.shape), dtype=np.uint8)
    _, tresh = cv.threshold(img_gray, treshold_number, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(tresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    cv.drawContours(black_image, contours, contour_index, (255,255,255),contour_stroke)
    converted_black_image = cv.cvtColor(black_image,cv.COLOR_GRAY2RGB)
    cv.imwrite(c_image_dir + c_image_contour_name + img_format, converted_black_image)
    ####
    invert = cv.bitwise_not(img_gray)
    blur = cv.GaussianBlur(invert, (21,21),0)
    invertedblur = cv.bitwise_not(blur)
    sketch = cv.divide(img_gray, invertedblur, scale=256.0)
    cv.imwrite(c_image_dir + c_image_sketch_name + img_format, sketch)

    img_k_means = c_image_dir + c_image_name + img_format
    img_contour = c_image_dir + c_image_contour_name + img_format
    img_sketch = c_image_dir + c_image_sketch_name + img_format
    
    return img_k_means, img_contour ,img_sketch

if __name__ == "__main__":
    app.run(debug=True)

