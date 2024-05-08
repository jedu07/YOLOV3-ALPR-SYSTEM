from app import ParkingProperties, db, app,VehicleDatabase, PlateRecognitionLog, Admin

#donna = VehicleDatabase('jed', '20@gmailc','STA5131E','car','student')
donna = Admin('admin','admin1234')


if __name__ == '__main__':
   with app.app_context():

    db.session.add(donna)
    db.session.commit()

