from app import ParkingProperties, db, app,VehicleDatabase, PlateRecognitionLog

#donna = VehicleDatabase('jed', '20@gmailc','STA5131E','car','student')
donna = ParkingProperties('san isidro', 20,20,0,0)


if __name__ == '__main__':
   with app.app_context():

    db.session.add(donna)
    db.session.commit()

