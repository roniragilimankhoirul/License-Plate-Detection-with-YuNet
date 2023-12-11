# demo.py
import argparse
import numpy as np
import cv2 as cv  # Add this line
import easyocr
from lpd_yunet import LPD_YuNet
from PIL import Image  # Add this line
import base64
import io
from database import save_to_database

# Check OpenCV version
assert cv.__version__ >= "4.8.0", \
       "Please install the latest opencv-python to try this demo: python3 -m pip install --upgrade opencv-python"

# Valid combinations of backends and targets
backend_target_pairs = [
    [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA_FP16],
    [cv.dnn.DNN_BACKEND_TIMVX,  cv.dnn.DNN_TARGET_NPU],
    [cv.dnn.DNN_BACKEND_CANN,   cv.dnn.DNN_TARGET_NPU]
]

parser = argparse.ArgumentParser(description='LPD-YuNet for License Plate Detection')
parser.add_argument('--input', '-i', type=str,
                    help='Usage: Set path to the input image. Omit for using the default camera.')
parser.add_argument('--model', '-m', type=str, default='license_plate_detection_lpd_yunet_2023mar.onnx',
                    help='Usage: Set model path, defaults to license_plate_detection_lpd_yunet_2023mar.onnx.')
parser.add_argument('--save', '-s', action='store_true',
                    help='Usage: Specify to save file with results (i.e. bounding box, confidence level). Invalid in case of camera input.')
parser.add_argument('--vis', '-v', action='store_true',
                    help='Usage: Specify to open a new window to show results. Invalid in case of camera input.')
args = parser.parse_args()

def visualize(image, dets, line_color=(0, 255, 0), text_color=(0, 0, 255), fps=None):
    output = image.copy()

    if fps is not None:
        cv.putText(output, 'FPS: {:.2f}'.format(fps), (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, text_color)

    for det in dets:
        bbox = det[:-1].astype(int)
        x1, y1, x2, y2, x3, y3, x4, y4 = bbox

        # Draw the border of the license plate
        cv.line(output, (x1, y1), (x2, y2), line_color, 2)
        cv.line(output, (x2, y2), (x3, y3), line_color, 2)
        cv.line(output, (x3, y3), (x4, y4), line_color, 2)
        cv.line(output, (x4, y4), (x1, y1), line_color, 2)

    return output

def extract_text(image, results, reader):
    for det in results:
        bbox = det[:-1].astype(np.int32)
        x1, y1, x2, y2, x3, y3, x4, y4 = bbox

        # Extract text using OCR
        plate_roi = image[y1:y3, x1:x3]

        # Check if the image is grayscale (for webcam input)
        if len(plate_roi.shape) == 2:
            plate_texts = reader.readtext(plate_roi)
        else:
            plate_texts = reader.readtext(plate_roi, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')  # Specify the allowed characters

        # Choose the most confident prediction
        if plate_texts:
            best_prediction = max(plate_texts, key=lambda x: x[1])
            plate_text = str(best_prediction[-2])[1:]  # Extracted text without the first character
            confidence = best_prediction[1]  # Confidence score

            # Print the extracted text and confidence for debugging
            print("Extracted Text:", plate_text)
            print("Confidence:", confidence)

            # Draw the border of the license plate
            cv.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.line(image, (x2, y2), (x3, y3), (0, 255, 0), 2)
            cv.line(image, (x3, y3), (x4, y4), (0, 255, 0), 2)
            cv.line(image, (x4, y4), (x1, y1), (0, 255, 0), 2)

            # Draw results on the input image
            font = cv.FONT_HERSHEY_SIMPLEX
            cv.putText(image, plate_text, (x1, y1 - 5), font, 0.8, (0, 255, 0), 2)

            # Convert the image to base64
            img_pil = Image.fromarray(image)
            img_byte_array = io.BytesIO()
            img_pil.save(img_byte_array, format='JPEG')
            img_base64 = base64.b64encode(img_byte_array.getvalue()).decode('utf-8')

            # Save the result to the database
            save_to_database(plate_text, image)  # Pass the image directly

    return image

if __name__ == '__main__':
    backend_id = backend_target_pairs[0][0]
    target_id = backend_target_pairs[0][1]

    # Instantiate LPD-YuNet
    model = LPD_YuNet(modelPath=args.model,
                      confThreshold=0.9,
                      nmsThreshold=0.3,
                      topK=5000,
                      keepTopK=750,
                      backendId=backend_id,
                      targetId=target_id)

    # Instantiate EasyOCR reader
    reader = easyocr.Reader(['en'], gpu=True)  # Specify the language(s) you want to support

    # If input is an image
    if args.input is not None:
        image = cv.imread(args.input)
        h, w, _ = image.shape

        # Inference
        model.setInputSize([w, h])
        results = model.infer(image)

        # Print results
        print('{} license plates detected.'.format(results.shape[0]))

        # Extract and draw text on the input image
        image = extract_text(image, results, reader)

        # Save results if save is true
        # if args.save:
        #     print('Results saved to result.jpg')
        #     cv.imwrite('result.jpg', image)

        # Visualize results in a new window
        if args.vis:
            cv.imshow("LPD-YuNet Demo", image)
            cv.waitKey(0)

    else:  # Omit input to call the default camera
        deviceId = 2
        cap = cv.VideoCapture(deviceId)
        w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        model.setInputSize([w, h])

        tm = cv.TickMeter()
        while cv.waitKey(1) < 0:
            hasFrame, frame = cap.read()
            if not hasFrame:
                print('No frames grabbed!')
                break

            # Inference
            tm.start()
            results = model.infer(frame)  # results is a tuple
            tm.stop()

            # Draw results on the input image
            frame = visualize(frame, results, fps=tm.getFPS())

            # Extract and draw text on the input image
            frame = extract_text(frame, results, reader)

            # Visualize results in a new Window
            cv.imshow('LPD-YuNet Demo', frame)

            tm.reset()
