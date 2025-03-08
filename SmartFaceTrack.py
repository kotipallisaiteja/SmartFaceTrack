import cv2
import numpy as np
import sqlite3
import os
from datetime import datetime

# Constants
DB_NAME = "faces.db"
ATTENDANCE_DB = "attendance.db"
MODEL_FILE = "trained_model.xml"
NAMES_FILE = "names.npy"
FACE_COUNT = 100  # Capture 100 images per person
IMAGE_SIZE = (300, 300)  # Increased image resolution
CONFIDENCE_THRESHOLD = 40  # Lowered confidence threshold for better recognition

# Initialize databases
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS faces (
                        id INTEGER PRIMARY KEY AUTOINCREMENT, 
                        name TEXT COLLATE NOCASE,  
                        image BLOB, 
                        timestamp TEXT)''')
    
    conn.commit()
    conn.close()

    conn = sqlite3.connect(ATTENDANCE_DB)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS attendance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT,
                        image BLOB,
                        date TEXT,
                        timestamp TEXT)''')
    conn.commit()
    conn.close()

# Convert image to binary
def image_to_binary(image):
    _, buffer = cv2.imencode(".jpg", image)
    return buffer.tobytes()

# Check if name already exists in attendance for today
def is_already_marked(name):
    conn = sqlite3.connect(ATTENDANCE_DB)
    cursor = conn.cursor()
    today_date = datetime.now().strftime("%Y-%m-%d")

    cursor.execute("SELECT COUNT(*) FROM attendance WHERE name=? AND date=?", (name, today_date))
    exists = cursor.fetchone()[0] > 0

    conn.close()
    return exists

# Save captured face to the database
# Save captured face to the database with limit check
def save_face(name, image):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM faces WHERE name=?", (name,))
    count = cursor.fetchone()[0]

    if count < FACE_COUNT:
        cursor.execute("INSERT INTO faces (name, image, timestamp) VALUES (?, ?, ?)",
                       (name, image_to_binary(image), datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        print(f"✅ Saved face {count + 1}/{FACE_COUNT} for: {name}")
    else:
        print(f"⚠ Maximum {FACE_COUNT} images already stored for {name}.")

    conn.close()


# Capture faces and store in the database
def capture_faces():
    name = input("Enter person's name: ").strip()

    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
    cap = cv2.VideoCapture(0)
    count = 0

    # Check if name already exists in the database
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM faces WHERE name=?", (name,))
    existing_entry = cursor.fetchone()
    conn.close()

    if existing_entry and count == 0:
        print(f"⚠ Name '{name}' is already stored in the database, adding more images.")  

    while count < FACE_COUNT:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture image.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, IMAGE_SIZE)
            face = cv2.equalizeHist(face)

            save_face(name, face)

            count += 1
            cv2.putText(frame, f"Captured: {count}/{FACE_COUNT}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Face Capture', frame)

        if cv2.waitKey(1) & 0xFF == 13 or count == FACE_COUNT:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Face collection completed for {name}!")

# Train face recognition model
def train_model():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM faces")
    records = cursor.fetchall()
    conn.close()

    if not records:
        print("❌ No images found in database! Capture faces first.")
        return None

    Training_Data, Labels, Names = [], [], {}
    unique_names = {}
    label_id = 0

    for row in records:
        name = row[1]
        if name not in unique_names:
            unique_names[name] = label_id
            label_id += 1

        nparr = np.frombuffer(row[2], np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, IMAGE_SIZE)
        img = cv2.equalizeHist(img)
        Training_Data.append(np.asarray(img, dtype=np.uint8))
        Labels.append(unique_names[name])

    Labels = np.asarray(Labels, dtype=np.int32)
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(np.asarray(Training_Data), np.asarray(Labels))
    model.save(MODEL_FILE)
    np.save(NAMES_FILE, unique_names)
    print("✅ Model training completed!")

# Save attendance to the database
def save_attendance(name, image):
    if is_already_marked(name):
        print(f"⚠ Attendance already recorded for {name} today.")
        return

    conn = sqlite3.connect(ATTENDANCE_DB)
    cursor = conn.cursor()
    today_date = datetime.now().strftime("%Y-%m-%d")

    cursor.execute("INSERT INTO attendance (name, image, date, timestamp) VALUES (?, ?, ?, ?)",
                   (name, image_to_binary(image), today_date, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    conn.commit()
    conn.close()
    print(f"✅ Attendance recorded for {name}.")

# Detect faces and log attendance
def detect_faces():
    if not os.path.exists(MODEL_FILE) or not os.path.exists(NAMES_FILE):
        print("❌ Error: No trained model found! Train the model first.")
        return

    model = cv2.face.LBPHFaceRecognizer_create()
    model.read(MODEL_FILE)
    Names = np.load(NAMES_FILE, allow_pickle=True).item()
    name_map = {v: k for k, v in Names.items()}
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

    cap = cv2.VideoCapture(0)
    detected_names = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, IMAGE_SIZE)
            face = cv2.equalizeHist(face)
            label, confidence = model.predict(face)

            if confidence < CONFIDENCE_THRESHOLD and label in name_map:
                name = name_map[label]
                text = f"{name} - {100 - confidence:.2f}% Conf."

                if name not in detected_names:
                    save_attendance(name, frame)
                    detected_names.add(name)
            else:
                text = "Unknown"

            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == 13:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    init_db()
    print("✅ Facial recognition system initialized!")
    option = input("Choose an option - (1) Capture Faces, (2) Train Model, (3) Detect Faces: ").strip()
    
    if option == "1":
        capture_faces()
    elif option == "2":
        train_model()
    elif option == "3":
        detect_faces()
    else:
        print("❌ Invalid choice! Exiting...")