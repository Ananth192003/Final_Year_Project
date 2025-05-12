from flask import Flask, request, jsonify, render_template, send_from_directory, url_for, redirect, session
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from skimage.feature import local_binary_pattern
import base64
import io
import os

# Get the absolute path to the static directory
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
app = Flask(__name__, 
           static_folder=static_dir,
           static_url_path='/static')

IMAGE_SIZE = (48, 48)
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load both models
try:
    model1 = load_model("models/Akka_aug.h5")
    model2 = load_model("models/ferNet_with_lbp.keras")
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {str(e)}")
    model1 = None
    model2 = None

def preprocess_single_from_pil(img):
    img = img.convert('L')
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    return np.expand_dims(img_array, axis=0)

def compute_lbp(image, P=8, R=1):
    image_uint8 = (image * 255).astype(np.uint8)
    lbp = local_binary_pattern(image_uint8[:, :, 0], P, R, method='uniform')
    lbp = np.expand_dims(lbp, axis=-1).astype(np.float32)
    if lbp.max() > 0:
        lbp /= lbp.max()
    return lbp

def preprocess_for_lbp_from_pil(img):
    img = img.convert('L')
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    raw_batch = np.expand_dims(img_array, axis=0)
    lbp_img = compute_lbp(img_array)
    lbp_batch = np.expand_dims(lbp_img, axis=0)
    return raw_batch, lbp_batch

@app.route("/")
def index():
    return redirect("/login")

@app.route("/login")
def login():
    return render_template("loginui.html")

@app.route("/home")
def home():
    return render_template("home_ui.html")

@app.route("/login", methods=["POST"])
def handle_login():
    # For now, we'll just redirect to home after login
    # In a real application, you would validate credentials here
    return redirect("/home")

@app.route("/admin")
def admin_dashboard():
    return render_template("adminDashboard.html")

# Quiz data for all subjects
QUIZ_DATA = {
    'math': {
        1: {
            'title': 'Math Quiz 1',
            'questions': [
                {
                    'question': 'What is the value of π (pi) to two decimal places?',
                    'options': ['3.14', '3.16', '3.12', '3.18'],
                    'correct': 0
                },
                {
                    'question': 'What is the square root of 144?',
                    'options': ['12', '14', '16', '18'],
                    'correct': 0
                },
                {
                    'question': 'What is the sum of the angles in a triangle?',
                    'options': ['90 degrees', '180 degrees', '270 degrees', '360 degrees'],
                    'correct': 1
                },
                {
                    'question': 'What is 2 to the power of 5?',
                    'options': ['16', '32', '64', '128'],
                    'correct': 1
                },
                {
                    'question': 'What is the area of a circle with radius 5?',
                    'options': ['25π', '50π', '75π', '100π'],
                    'correct': 0
                }
            ]
        },
        2: {
            'title': 'Math Quiz 2',
            'questions': [
                {
                    'question': 'What is the derivative of x²?',
                    'options': ['x', '2x', 'x²', '2x²'],
                    'correct': 1
                },
                {
                    'question': 'What is the value of log₁₀(100)?',
                    'options': ['1', '2', '3', '4'],
                    'correct': 1
                },
                {
                    'question': 'What is the slope of the line y = 2x + 3?',
                    'options': ['2', '3', '5', '6'],
                    'correct': 0
                },
                {
                    'question': 'What is the sum of the first 10 natural numbers?',
                    'options': ['45', '50', '55', '60'],
                    'correct': 2
                },
                {
                    'question': 'What is the value of sin(90°)?',
                    'options': ['0', '0.5', '1', '2'],
                    'correct': 2
                }
            ]
        }
    },
    'physics': {
        1: {
            'title': 'Physics Quiz 1',
            'questions': [
                {
                    'question': 'What is the SI unit of force?',
                    'options': ['Newton', 'Joule', 'Watt', 'Pascal'],
                    'correct': 0
                },
                {
                    'question': 'Which of these is a vector quantity?',
                    'options': ['Speed', 'Distance', 'Velocity', 'Temperature'],
                    'correct': 2
                },
                {
                    'question': 'What is the acceleration due to gravity on Earth?',
                    'options': ['9.8 m/s²', '10 m/s²', '8.9 m/s²', '11 m/s²'],
                    'correct': 0
                },
                {
                    'question': 'Which law states that every action has an equal and opposite reaction?',
                    'options': ["Newton's First Law", "Newton's Second Law", "Newton's Third Law", "Law of Conservation of Energy"],
                    'correct': 2
                },
                {
                    'question': 'What is the unit of electric current?',
                    'options': ['Volt', 'Ampere', 'Ohm', 'Watt'],
                    'correct': 1
                }
            ]
        },
        2: {
            'title': 'Physics Quiz 2',
            'questions': [
                {
                    'question': 'Which of these is NOT a type of electromagnetic wave?',
                    'options': ['Radio waves', 'Sound waves', 'X-rays', 'Gamma rays'],
                    'correct': 1
                },
                {
                    'question': 'What is the speed of light in a vacuum?',
                    'options': ['300,000 km/s', '299,792 km/s', '300,000 m/s', '299,792 m/s'],
                    'correct': 1
                },
                {
                    'question': 'Which of these is a scalar quantity?',
                    'options': ['Force', 'Velocity', 'Acceleration', 'Speed'],
                    'correct': 3
                },
                {
                    'question': 'What is the unit of power?',
                    'options': ['Joule', 'Watt', 'Newton', 'Pascal'],
                    'correct': 1
                },
                {
                    'question': 'Which law states that the pressure of a gas is inversely proportional to its volume at constant temperature?',
                    'options': ["Boyle's Law", "Charles's Law", "Gay-Lussac's Law", "Avogadro's Law"],
                    'correct': 0
                }
            ]
        }
    },
    'chemistry': {
        1: {
            'title': 'Chemistry Quiz 1',
            'questions': [
                {
                    'question': 'What is the basic unit of matter?',
                    'options': ['Atom', 'Molecule', 'Cell', 'Element'],
                    'correct': 0
                },
                {
                    'question': 'Which of these is NOT a state of matter?',
                    'options': ['Solid', 'Liquid', 'Gas', 'Energy'],
                    'correct': 3
                },
                {
                    'question': 'What is the chemical symbol for gold?',
                    'options': ['Ag', 'Au', 'Fe', 'Cu'],
                    'correct': 1
                },
                {
                    'question': 'What is the pH value of pure water?',
                    'options': ['5', '7', '9', '14'],
                    'correct': 1
                },
                {
                    'question': 'Which element is the most abundant in the Earth\'s atmosphere?',
                    'options': ['Oxygen', 'Carbon', 'Nitrogen', 'Hydrogen'],
                    'correct': 2
                }
            ]
        },
        2: {
            'title': 'Chemistry Quiz 2',
            'questions': [
                {
                    'question': 'What is the process called when a solid turns directly into a gas?',
                    'options': ['Evaporation', 'Condensation', 'Sublimation', 'Melting'],
                    'correct': 2
                },
                {
                    'question': 'Which of these is a noble gas?',
                    'options': ['Oxygen', 'Nitrogen', 'Helium', 'Chlorine'],
                    'correct': 2
                },
                {
                    'question': 'What is the chemical formula for water?',
                    'options': ['H2O', 'CO2', 'O2', 'H2'],
                    'correct': 0
                },
                {
                    'question': 'Which of these is an example of a chemical change?',
                    'options': ['Melting ice', 'Boiling water', 'Burning wood', 'Breaking glass'],
                    'correct': 2
                },
                {
                    'question': 'What is the atomic number of carbon?',
                    'options': ['6', '12', '14', '16'],
                    'correct': 0
                }
            ]
        }
    }
}

# Quiz routes for all subjects
@app.route("/quiz/<subject>/<int:quiz_number>")
def show_quiz(subject, quiz_number):
    if subject not in QUIZ_DATA or quiz_number not in QUIZ_DATA[subject]:
        return redirect("/home")
    return render_template(f"{subject}.html", quiz_data=QUIZ_DATA[subject][quiz_number])

@app.route("/math")
def math():
    return render_template("math.html", quiz_data=QUIZ_DATA['math'])

@app.route("/physics")
def physics():
    return render_template("physics.html", quiz_data=QUIZ_DATA['physics'])

@app.route("/chemistry")
def chemistry():
    return render_template("chemistry.html", quiz_data=QUIZ_DATA['chemistry'])

@app.route("/chemistry/<int:module_number>")
def show_chemistry_module(module_number):
    return render_template("chemistry.html", module_number=module_number, quiz_data=QUIZ_DATA['chemistry'])

@app.route("/module/<int:module_number>")
def show_module(module_number):
    return render_template("module.html", module_number=module_number)

@app.route("/static/pdfs/<path:filename>")
def serve_pdf(filename):
    try:
        # If physics PDF is requested but doesn't exist, try using the math PDF
        if filename.startswith('physics_module'):
            module_num = filename.split('module')[1].split('.')[0]
            alternate_filename = f"module{module_num}.pdf"
            try:
                return send_from_directory("static/pdfs", alternate_filename)
            except:
                return "Module content is being prepared. Please check back later.", 404
        return send_from_directory("static/pdfs", filename)
    except Exception as e:
        print(f"Error serving PDF {filename}: {str(e)}")
        return "Module content is being prepared. Please check back later.", 404

@app.route("/predict", methods=["POST"])
def predict():
    if model1 is None or model2 is None:
        return jsonify({"error": "Models not loaded. Please check the models directory."}), 500

    try:
        data = request.json['image']
        image_data = base64.b64decode(data.split(',')[1])
        img = Image.open(io.BytesIO(image_data)).convert('RGB')

        img_model1 = preprocess_single_from_pil(img)
        raw_batch, lbp_batch = preprocess_for_lbp_from_pil(img)

        pred1 = model1.predict(img_model1)
        pred2 = model2.predict([raw_batch, lbp_batch])

        combined = 0.6 * pred1 + 0.4 * pred2
        predicted_index = np.argmax(combined, axis=1)[0]
        predicted_emotion = class_names[predicted_index]
        confidence = float(combined[0][predicted_index]) * 100

        print(f"Predicted emotion: {predicted_emotion} with confidence: {confidence:.2f}%")

        return jsonify({
            "emotion": predicted_emotion,
            "confidence": round(confidence, 2)
        })
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/test-images")
def test_images():
    return render_template("image_test.html")

@app.route("/static/images/<path:filename>")
def serve_image(filename):
    try:
        return send_from_directory("static/images", filename)
    except Exception as e:
        print(f"Error serving image {filename}: {str(e)}")
        return f"Error: Image {filename} not found", 404

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("static/pdfs", exist_ok=True)
    os.makedirs("static/images", exist_ok=True)
    
    # Check if models directory exists
    if not os.path.exists("models"):
        print("Warning: 'models' directory not found. Please create it and add the model files.")
    
    # Check if PDFs directory is empty
    pdf_dir = "static/pdfs"
    if not os.listdir(pdf_dir):
        print(f"Warning: No PDFs found in {pdf_dir}. Please add your module PDFs.")
    
    print("Starting Smart Learning System...")
    app.run(debug=True, port=5000)