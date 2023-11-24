import streamlit as st
import numpy as np
import cv2
from pyzbar.pyzbar import decode
from st_clickable_images import clickable_images
import base64
import requests
from datetime import datetime
from bridge_wrapper import YOLOv7_DeepSORT, Detector

detector = Detector(iou_thresh=0.6, conf_thres=0.3)  # it'll detect ONLY [person,horses,sports ball]. class = None means detect all classes. List info at: "data/coco.yaml"
detector.load_model('./weights/best2.pt')  # pass the path to the trained weight file
# Initialise  class that binds detector and tracker in one class
tracker = YOLOv7_DeepSORT(reID_model_path="./deep_sort/model_weights/mars-small128.pb", detector=detector)

def label_screen():
    st.subheader("Barcode: ", st.session_state.code)
    # output = None will not save the output video
    tracker.track_video(0, output=None, show_live=True, skip_frames=0, count_objects=True, verbose=1)


def camera_screen():
    camera_placeholder = st.empty()

    picture = camera_placeholder.camera_input("Take a picture of the parcel tracking barcode!", key="picture")

    if st.session_state.picture is not None:
        bytes_data = st.session_state.picture.getvalue()
        image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        bd = cv2.barcode.BarcodeDetector()
        detectedBarcodes = decode(image)

        if not detectedBarcodes:
            st.write("Barcode Not Detected or your barcode is blank/corrupted!")
            code = st.text_input("Enter Barcode Manually")
        else:
            camera_placeholder.empty()
            code = ""
            for barcode in detectedBarcodes:

                (x, y, w, h) = barcode.rect
                cv2.rectangle(image, (x - 10, y - 10),
                              (x + w + 10, y + h + 10),
                              (255, 0, 0), 2)
                if barcode.data != "":
                    code = barcode.data.decode()

            # Now do something with the image! For example, let's display it:
            st.image(image, channels="BGR")
            st.subheader(code)
            if st.button("Retake Picture"):
                del st.session_state.picture
                st.rerun()
        if st.button("Deliver Parcel"):
            if code == "":
                st.write("Please enter/scan barcode first!")
            else:
                st.session_state.code = code
                code = "https://id.gs1.org/01/" + code
                message = {
                        "@context": "https://ref.gs1.org/standards/epcis/2.0.0/epcis-context.jsonld",
                        "type": "TransactionEvent",
                        "eventTime": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f+09:00"),
                        "eventTimeZoneOffset": "+09:00",
                        "bizTransactionList": [
                            {
                                "type": "po",
                                "bizTransaction": "https://id.gs1.org/253/06141410000121618034"
                            }
                        ],
                        "parentID": "https://id.gs1.org/00/106141412345678908",
                        "epcList": [
                            code
                        ],
                        "action": "ADD",
                        "bizStep": "receiving"
                    }
                url = "http://143.248.219.189:8090/epcis/v2/events"
                r = requests.post(url, json=message)
                print(r.status_code)
                print(r.text)
                st.rerun()


if __name__ == '__main__':
    try:
        label_screen()
    except AttributeError:
        camera_screen()