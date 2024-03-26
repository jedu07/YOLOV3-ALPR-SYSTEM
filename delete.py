from app import PlateRecognitionLog, db, app,ParkingProperties



if __name__ == '__main__':
   with app.app_context():
    db.session.query(ParkingProperties).filter(ParkingProperties.id == '1').delete()
    db.session.commit()

