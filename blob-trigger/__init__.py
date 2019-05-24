import logging

import azure.functions as func
import cv2
import numpy as np

def main(myblob: func.InputStream, outputblob: func.Out[func.InputStream]):
    logging.info(f"Python blob trigger function processed blob \n"
                 f"Name: {myblob.name}\n"
                 f"Blob Size: {myblob.length} bytes")

    img_buffer = myblob.read()    
    img_data = np.frombuffer(img_buffer, dtype=np.uint8)
    image_result = blur_function(img_data)
    
    result = cv2.imencode(".jpg", image_result)[1]
    data_encode = np.array(result)
    outputblob.set(data_encode.tobytes())

def blur_function(img):
    img = cv2.imdecode(img, -1)

    kernel = np.ones((5,5),np.float32)/25
    dst = cv2.filter2D(img,-1,kernel)

    return dst
