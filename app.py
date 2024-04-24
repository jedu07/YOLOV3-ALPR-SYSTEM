from flask import Flask, request, flash, url_for, redirect, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_login import login_user, login_required, logout_user, current_user, UserMixin, LoginManager
from sqlalchemy.sql import func
from werkzeug.security import generate_password_hash, check_password_hash
from flask import jsonify

app = Flask(__name__)

app.secret_key = "secret key"

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db?check_same_thread=False'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'secret key'

db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)


@login_manager.user_loader
def load_user(id):
    return Admin.query.get(int(id))


class Admin(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    user_name = db.Column(db.String(100), nullable=False, unique=True)
    password = db.Column(db.String(100), nullable=False)

    def __init__(self, user_name, password):
        self.user_name = user_name
        self.password = password

    def __repr__(self):
        return '<Admin %r' % self.id


class ParkingProperties(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    parking_name = db.Column(db.String(100), nullable=False, unique=True)
    max_slots_car = db.Column(db.Integer, nullable=False)
    max_slots_motor = db.Column(db.Integer, nullable=False)
    current_slots_car = db.Column(db.Integer, nullable=True)
    current_slots_motor = db.Column(db.Integer, nullable=True)

    def __init__(self, parking_name, max_slots_car, max_slots_motor, current_slots_car, current_slots_motor):
        self.parking_name = parking_name
        self.max_slots_car = max_slots_car
        self.max_slots_motor = max_slots_motor
        self.current_slots_car = current_slots_car
        self.current_slots_motor = current_slots_motor

    def __repr__(self):
        return '<parking_properties %r' % self.id


class VehicleDatabase(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    num_plate = db.Column(db.String(100), nullable=False, unique=True)
    type = db.Column(db.String(100), nullable=False, default="")
    identity = db.Column(db.String(100), nullable=False, default="")

    def __init__(self, name, email, num_plate, type, identity):
        self.name = name
        self.email = email
        self.num_plate = num_plate
        self.type = type
        self.identity = identity

    def __repr__(self):
        return '<Vehicle_DB %r' % self.id


class PlateRecognitionLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    owner_name = db.Column(db.String(100), nullable=False)
    plate_num = db.Column(db.String, db.ForeignKey('vehicle_database.num_plate'))
    in_out = db.Column(db.String(100), default='out')
    date = db.Column(db.DateTime(timezone=True), default=func.now())
    vehicle_database = db.relationship('VehicleDatabase', backref='plate_recognition_logs', lazy=True)

    def __init__(self, plate_num, in_out, owner_name):
        self.owner_name = owner_name
        self.plate_num = plate_num
        self.in_out = in_out

    def __repr__(self):
        return '<plate_recognition %r' % self.id


## Routes ##
@app.route('/login', methods=['GET', 'POST'])
@app.route('/', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        user_ID = request.form.get('user_name')
        password = request.form.get('password')

        user = Admin.query.filter_by(user_name=user_ID).first()
        if user:
            if check_password_hash(user.password, password):
                flash('Logged in successfully!', category='success')
                login_user(user, remember=True)
                return redirect(url_for('dashboard'))
            else:
                error = 'Incorrect password, try again.'
        else:
            error = 'Username does not exist.'

    return render_template("login_page.html", user=current_user, error=error)


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.route('/f', methods=['GET', 'POST'])
@login_required
def list_authors():
    authors = PlateRecognitionLog.query.all()
    return render_template('list_authors.html', authors=authors, user=current_user)





@app.route('/registration', methods=['GET', 'POST'])
@login_required
def add_plate():
    if request.method == 'POST':
        if not request.form['name'] or not request.form['email'] or not request.form['plate'] or not request.form[
            'vehicle_type'] or not request.form['identity_type']:
            flash('please enter all the fields', 'error')
        else:
            newPlate = VehicleDatabase(request.form['name'], request.form['email'], request.form['plate'],
                                       request.form['vehicle_type'], request.form['identity_type'])

            db.session.add(newPlate)
            db.session.commit()
            flash('all goods')
            return redirect(url_for('dashboard'))
    return render_template('registration.html')


# Route to edit a registered vehicle

@app.route('/edit/<int:id>', methods=['GET', 'POST'])
##@login_required
def edit_plate(id):
    vehicle = VehicleDatabase.query.get_or_404(id)  # Get vehicle by ID or return 404 error

    if request.method == 'POST':
        if not request.form['name'] or not request.form['email'] or not request.form['plate'] or not request.form['vehicle_type'] or not request.form['identity_type']:
            flash('Please enter all the fields', 'error')
        else:
            vehicle.name = request.form['name']
            vehicle.email = request.form['email']
            vehicle.num_plate = request.form['plate']
            vehicle.type = request.form['vehicle_type']
            vehicle.identity = request.form['identity_type']

            db.session.commit()
            flash('Vehicle details updated successfully!', 'success')
            return redirect(url_for('dashboard'))

    return render_template('edit.html', vehicle=vehicle)



@app.route('/delete/<int:id>', methods=['POST'])
##@login_required
def delete_plate(id):
    try:
        vehicle = VehicleDatabase.query.get_or_404(id)
        db.session.delete(vehicle)
        db.session.commit()
        return jsonify({"status": "success", "message": "Plate number deleted successfully."})
    except Exception as e:
        db.session.rollback()
        return jsonify({"status": "error", "message": "Error deleting plate number."})



@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    # Get vehicle entering premises data
    entries = PlateRecognitionLog.query.all()

    # Get current parking slot status
    parking_properties = ParkingProperties.query.first()
    car_slots = f"{parking_properties.current_slots_car}/{parking_properties.max_slots_car}" if parking_properties else "N/A"
    motor_slots = f"{parking_properties.current_slots_motor}/{parking_properties.max_slots_motor}" if parking_properties else "N/A"

    return render_template('dashboard.html', entries=entries, car_slots=car_slots, motor_slots=motor_slots,
                           plates=VehicleDatabase.query.all())


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        db.session.commit()

    app.run(debug=True)
