**Overview**
  This project implements an automated attendance system using Face Recognition. It captures faces, trains a model, and detects faces in real-time to log attendance. The system enhances accuracy, eliminates         manual errors, and improves efficiency in attendance tracking.

**Key Features**
  Detects and recognizes faces in real-time
  Stores images and attendance records in a database
  Uses OpenCV’s LBPH algorithm for face recognition
  Prevents duplicate attendance entries
  Works efficiently under different lighting conditions

**Tech Stack**
  Programming Language: Python 
  Libraries: OpenCV, NumPy, SQLite3
  Algorithm: Local Binary Patterns Histogram (LBPH)
  Database: SQLite

**System Workflow**
  Face Capture → Captures face images and stores them in a database.
  Model Training → Trains an LBPH-based recognition model using stored images.
  Face Detection & Recognition → Detects faces in real-time and matches them with stored data.
  Attendance Logging → Marks attendance if a match is found and prevents duplicate entries.

**Installation & Usage**
  1. Install Dependencies
      Ensure you have Python 3.7+ installed. Run the following command:
      pip install opencv-python numpy sqlite3

  2. For capturing Faces, Training, Detecting and taking Attendence log
      python SmartFaceTrack
     
**Advantages**
  Fully automated attendance system
  Reduces manual effort and human errors
  Ensures accuracy and prevents duplicate attendance
  Can be integrated into schools, offices, and organizations

**Future Enhancements**
  Upgrade to deep learning models for improved accuracy 
  Integrate with cloud databases
  Implement multi-face detection & recognition in one frame

**Connect With Me**
  LinkedIn Profile: https://www.linkedin.com/in/sai-teja-kotipalli-8b83b7259
  Email: kotipallisaiteja@gmail.com
