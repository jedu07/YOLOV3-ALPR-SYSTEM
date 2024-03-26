from app import Author, db, app


if __name__ == '__main__':
   with app.app_context():
    update = Author.query.filter_by(name='fuck').first()

    update.title = 'erere444r'



    db.session.commit()

