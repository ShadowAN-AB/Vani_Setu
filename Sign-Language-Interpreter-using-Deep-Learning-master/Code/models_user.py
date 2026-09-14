"""SQLite (default) or Postgres user accounts for email/password login."""

import os
from datetime import datetime, timezone

from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash

db = SQLAlchemy()


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def to_public(self):
        return {"id": self.id, "email": self.email}


def database_uri(app_dir):
    """Use DATABASE_URL on a host (Postgres). Otherwise a local SQLite file."""
    url = os.environ.get("DATABASE_URL", "").strip()
    if url:
        # Render/Heroku sometimes give postgres:// which SQLAlchemy rejects
        if url.startswith("postgres://"):
            url = "postgresql://" + url[len("postgres://"):]
        return url
    sqlite_path = os.path.join(app_dir, "users.db")
    return "sqlite:///" + sqlite_path


def init_db(app):
    app.config["SQLALCHEMY_DATABASE_URI"] = database_uri(app.static_folder or os.path.dirname(os.path.abspath(__file__)))
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    db.init_app(app)
    with app.app_context():
        db.create_all()
