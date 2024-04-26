from app import ParkingProperties, db, app

if __name__ == '__main__':
    with app.app_context():
        # Delete all entries from PlateRecognitionLog
       # db.session.query(PlateRecognitionLog).delete()

        # Set current_slots_car to 0 in ParkingProperties
        parking_properties = ParkingProperties.query.first()
        if parking_properties:
            parking_properties.current_slots_car = 0

        db.session.commit()
