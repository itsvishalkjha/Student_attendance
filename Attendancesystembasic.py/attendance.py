import cv2
import mediapipe as mp
import pandas as pd
from datetime import datetime

# Initialize face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize student tracking
students = {}
student_ids = 1

# Initialize the Excel file
excel_file = 'student_presence.xlsx'
try:
    df = pd.read_excel(excel_file, index_col=0)
except FileNotFoundError:
    df = pd.DataFrame(columns=["Student", "First_Seen"])

# Function to detect faces and record the time
def detect_faces_and_record_time(image, gray):
    global student_ids

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_id = None
        for key, (coords, _) in students.items():
            if abs(coords[0] - x) < 30 and abs(coords[1] - y) < 30:
                face_id = key
                break

        if face_id is None:
            face_id = f'Student{student_ids}'
            students[face_id] = ((x, y, w, h), datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            student_ids += 1

        students[face_id] = ((x, y, w, h), students[face_id][1])

        if face_id not in df.index:
            df.loc[face_id] = [face_id, students[face_id][1]]

        # Draw rectangle around the face and label
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(image, f'{face_id} present', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return image

# Video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = detect_faces_and_record_time(image, gray)

    cv2.imshow('Student Presence Detection', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Save the data to Excel file
df.to_excel(excel_file)

cap.release()
cv2.destroyAllWindows()
