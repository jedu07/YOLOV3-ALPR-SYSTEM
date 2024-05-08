import pytesseract
import cv2
import numpy as np
import re
from app import VehicleDatabase, db, app, PlateRecognitionLog, ParkingProperties
import os
# If you don't have tesseract executable in your PATH, include the following:
# pytesseract.pytesseract.tesseract_cmd = r'<full_path_to_your_tesseract_executable>'
# Example tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract'

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

def recognize_license_plate():
    image_path = "detections/crop/0/license-plate_1.png"

    try:
        # Load the license plate image
        gray = cv2.imread(image_path, 0)

        if gray is None:
            raise FileNotFoundError("Failed to read image:", image_path)

        gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        gray = cv2.medianBlur(gray, 3)
        # perform otsu thresh (using binary inverse since opencv contours work better with white text)
        ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        # cv2.imshow("Otsu", thresh)
        # cv2.waitKey(0)
        rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # apply dilation
        dilation = cv2.dilate(thresh, rect_kern, iterations=1)
        # cv2.imshow("dilation", dilation)
        # cv2.waitKey(0)
        # find contours
        try:
            contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        except:
            ret_img, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

        # create copy of image
        im2 = gray.copy()

        plate_num = ""
        # loop through contours and find letters in license plate
        for cnt in sorted_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            height, width = im2.shape
            # if height of box is not tall enough relative to total height then skip
            if height / float(h) > 6: continue
            ratio = h / float(w)
            # if height to width ratio is less than 1.5 skip
            if ratio < 1.5: continue
            # if width is not wide enough relative to total width then skip
            if width / float(w) > 15: continue
            area = h * w
            # if area is less than 100 pixels skip
            if area < 100: continue

            # draw the rectangle
            rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # grab character region of image
            roi = thresh[y - 5:y + h + 5, x - 5:x + w + 5]
            # perfrom bitwise not to flip image to black text on white background
            roi = cv2.bitwise_not(roi)
            # perform another blur on character region
            roi = cv2.medianBlur(roi, 5)

            try:
                text = pytesseract.image_to_string(roi,
                                                   config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
                # clean tesseract text by removing any unwanted blank spaces
                clean_text = re.sub('[\W_]+', '', text)
                print(clean_text)
                plate_num += clean_text
            except:
                text = None
        if plate_num != None:
            print("License Plate #: ", plate_num)



        if plate_num != "":
            print("License Plate # : ", plate_num)
            os.remove("detections/crop/0/license-plate_1.png")
            with app.app_context():
                # Check if the plate exists in the VehicleDatabase
                exists = VehicleDatabase.query.filter_by(num_plate=plate_num).first() is not None
                print(exists)

                if exists:
                    # Query owner of the vehicle
                    query_database = VehicleDatabase.query.filter_by(num_plate=plate_num).first()
                    owner_name = query_database.name

                    # Get the vehicle type
                    vehicle_type = VehicleDatabase.query.filter_by(num_plate=plate_num).first().type

                    # Get the latest status of the plate (in or out of the parking)
                    latest_status = PlateRecognitionLog.query.filter_by(plate_num=plate_num).order_by(
                        PlateRecognitionLog.id.desc()).first()
                    status = latest_status.in_out if latest_status else None

                    if vehicle_type == 'car':
                        # Get the current and max slots for cars
                        parking_properties = ParkingProperties.query.first()
                        current_slots = parking_properties.current_slots_car
                        max_slots = parking_properties.max_slots_car

                        if status == 'out' or status is None:
                            # Plate detected and not inside parking
                            if current_slots < max_slots:
                                record = PlateRecognitionLog(owner_name=owner_name, plate_num=plate_num, in_out='in')
                                db.session.add(record)

                                # Update VehicleDatabase status to '1' (In)
                                vehicle = VehicleDatabase.query.filter_by(num_plate=plate_num).first()
                                if vehicle:
                                    vehicle.status = '1'

                                parking_properties.current_slots_car += 1
                                print('The plate is detected and will go in')

                        elif status == 'in':
                            # Plate detected and is inside the parking
                            record = PlateRecognitionLog(owner_name=owner_name, plate_num=plate_num, in_out='out')
                            db.session.add(record)

                            # Update VehicleDatabase status to '0' (Out)
                            vehicle = VehicleDatabase.query.filter_by(num_plate=plate_num).first()
                            if vehicle:
                                vehicle.status = '0'

                            parking_properties.current_slots_car -= 1
                            print('The plate is valid and will go out')
                        else:
                            print('Invalid plate status')

                    elif vehicle_type == 'motor':
                        # Get the current and max slots for cars
                        parking_properties = ParkingProperties.query.first()
                        current_slots = parking_properties.current_slots_car
                        max_slots = parking_properties.max_slots_car

                        if status == 'out' or status is None:
                            # Plate detected and not inside parking
                            if current_slots < max_slots:
                                record = PlateRecognitionLog(owner_name=owner_name, plate_num=plate_num, in_out='in')
                                db.session.add(record)

                                # Update VehicleDatabase status to '1' (In)
                                vehicle = VehicleDatabase.query.filter_by(num_plate=plate_num).first()
                                if vehicle:
                                    vehicle.status = '1'

                                parking_properties.current_slots_car += 1
                                print('The plate is detected and will go in')
                            else:
                                print('Parking is full')

                        elif status == 'in':
                            # Plate detected and is inside the parking
                            record = PlateRecognitionLog(owner_name=owner_name, plate_num=plate_num, in_out='out')
                            db.session.add(record)

                            # Update VehicleDatabase status to '0' (Out)
                            vehicle = VehicleDatabase.query.filter_by(num_plate=plate_num).first()
                            if vehicle:
                                vehicle.status = '0'

                            parking_properties.current_slots_car -= 1
                            print('The plate is valid and will go out')
                        else:
                            print('Invalid plate status')

                    else:
                        print('Invalid vehicle type')

                    db.session.commit()
                else:
                    print('Plate does not exist in the database')

    except FileNotFoundError as e:
        print("Error:", e)
        print("Skipping image processing for:", image_path)
        # Optionally, you can add code here to handle the case where the image is not found
