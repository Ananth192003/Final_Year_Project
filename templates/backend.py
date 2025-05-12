from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io
from tensorflow.keras.models import load_model
from skimage.feature import local_binary_pattern

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load emotion detection models
IMAGE_SIZE = (48, 48)
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
model1 = load_model("finally/models/Akka_aug.h5")
model2 = load_model("finally/models/ferNet_with_lbp.keras")

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

@app.post("/detect-emotion")
async def detect_emotion(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        
        img_model1 = preprocess_single_from_pil(img)
        raw_batch, lbp_batch = preprocess_for_lbp_from_pil(img)
        
        pred1 = model1.predict(img_model1)
        pred2 = model2.predict([raw_batch, lbp_batch])
        
        combined = 0.6 * pred1 + 0.4 * pred2
        predicted_index = np.argmax(combined, axis=1)[0]
        predicted_emotion = class_names[predicted_index]
        confidence = float(combined[0][predicted_index]) * 100
        
        return {
            "emotion": predicted_emotion,
            "confidence": round(confidence, 2)
        }
    except Exception as e:
        return {"error": str(e)}

questions = [
    {"question": "What is the atomic number of Hydrogen?", "options": ["1", "2", "3", "4"], "correct": "1"},
    {"question": "Which gas is most abundant in Earth's atmosphere?", "options": ["Oxygen", "Nitrogen", "Carbon Dioxide", "Argon"], "correct": "Nitrogen"},
    {"question": "What is the chemical formula of water?", "options": ["H2O", "CO2", "O2", "H2"], "correct": "H2O"},
    {"question": "What is the speed of light in vacuum?", "options": ["3x10^8 m/s", "3x10^6 m/s", "3x10^4 m/s", "3x10^2 m/s"], "correct": "3x10^8 m/s"},
    {"question": "What is the pH of pure water?", "options": ["7", "5", "9", "2"], "correct": "7"},
    {"question": "Which element is used in nuclear reactors as fuel?", "options": ["Uranium", "Thorium", "Plutonium", "Radium"], "correct": "Uranium"},
    {"question": "What is the chemical symbol for Gold?", "options": ["Ag", "Au", "Pb", "Fe"], "correct": "Au"},
    {"question": "Which type of bond is present in NaCl?", "options": ["Covalent", "Ionic", "Metallic", "Hydrogen"], "correct": "Ionic"},
    {"question": "What is the main component of natural gas?", "options": ["Methane", "Ethane", "Propane", "Butane"], "correct": "Methane"},
    {"question": "Which metal is the best conductor of electricity?", "options": ["Copper", "Silver", "Gold", "Aluminum"], "correct": "Silver"}
]

session_state = {"current_question_index": 0}

@app.get("/get-question")
async def get_question():
    """Fetches the next question."""
    index = session_state["current_question_index"]
    
    if index < len(questions):
        return questions[index]
    else:
        return {"message": "Quiz completed!"}  

@app.post("/next-question")
async def next_question():
    """Moves to the next question."""
    if session_state["current_question_index"] < len(questions) - 1:
        session_state["current_question_index"] += 1
        return {"message": "Next question loaded"}
    else:
        return {"message": "Quiz completed!"}

@app.get("/reset-quiz")
async def reset_quiz():
    """Resets the quiz progress."""
    session_state["current_question_index"] = 0
    return {"message": "Quiz has been reset!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5001, reload=True)
