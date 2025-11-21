from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
import os, pickle, nltk, string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.exceptions import NotFittedError

# Initialize app
app = Flask(__name__)
app.secret_key = "supersecretkey"  # Change in production
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Initialize database and login manager
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

# ----------- USER MODEL -----------
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(256))
    provider = db.Column(db.String(50), default="local")

# Create DB if not exists
with app.app_context():
    db.create_all()

# ----------- LOGIN MANAGER -----------
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# ----------- LOAD MODEL -----------
base_path = os.path.dirname(os.path.abspath(__file__))
vectorizer = None
model = None
try:
    with open(os.path.join(base_path, "vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)
except FileNotFoundError:
    vectorizer = None

try:
    with open(os.path.join(base_path, "model.pkl"), "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = None

# ----------- NLP SETUP -----------
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    y = [ps.stem(i) for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    return " ".join(y)

# ----------- ROUTES -----------

@app.route("/")
@login_required
def home():
    return render_template("index.html", username=current_user.email)

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash("Email already registered.")
            return redirect(url_for("register"))

        hashed_pw = generate_password_hash(password)
        new_user = User(email=email, password=hashed_pw)
        db.session.add(new_user)
        db.session.commit()
        flash("✅ Registration successful! Please log in.")
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            flash("✅ Logged in successfully!")
            return redirect(url_for("home"))
        else:
            flash("❌ Invalid credentials.")
    return render_template("login.html")

@app.route("/logout")
def logout():
    logout_user()
    flash("👋 Logged out.")
    return redirect(url_for("login"))

@app.route("/predict", methods=["POST"])
@login_required
def predict():
    input_sms = request.form.get("message", "").strip()
    if input_sms == "":
        flash("⚠️ Please enter a message.")
        return redirect(url_for("home"))

    if model is None or vectorizer is None:
        flash("⚠️ Model or vectorizer not found. Please run the training script.", "error")
        return redirect(url_for("home"))

    transformed_sms = transform_text(input_sms)
    try:
        vector_input = vectorizer.transform([transformed_sms])
        result = model.predict(vector_input)[0]
        if int(result) == 1:
            flash("🚨 Spam message detected!", "error")
        else:
            flash("✅ This message is not spam.", "success")
    except NotFittedError:
        flash("⚠️ Model not fitted. Check your model file.", "error")
    except Exception as e:
        flash(f"An error occurred during prediction: {e}", "error")

    return redirect(url_for("home"))

if __name__ == "__main__":
    app.run(debug=True)
