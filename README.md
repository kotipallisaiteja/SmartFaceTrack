**Overview** <br>
  This project implements an automated attendance system using Face Recognition. It captures faces, trains a model, and detects faces in real-time to log attendance. The system enhances accuracy, eliminates         manual errors, and improves efficiency in attendance tracking.

**Key Features** <br>
  Detects and recognizes faces in real-time<br>
  Stores images and attendance records in a database<br>
  Uses OpenCV’s LBPH algorithm for face recognition<br>
  Prevents duplicate attendance entries<br>
  Works efficiently under different lighting conditions

**Tech Stack** <br>
  Programming Language: Python <br>
  Libraries: OpenCV, NumPy, SQLite3 <br>
  Algorithm: Local Binary Patterns Histogram (LBPH) <br>
  Database: SQLite

**System Workflow** <br>
  Face Capture → Captures face images and stores them in a database. <br>
  Model Training → Trains an LBPH-based recognition model using stored images. <br>
  Face Detection & Recognition → Detects faces in real-time and matches them with stored data. <br>
  Attendance Logging → Marks attendance if a match is found and prevents duplicate entries.

**Installation & Usage**
  1. Install Dependencies
      Ensure you have Python 3.7+ installed. Run the following command:
      **pip install opencv-python numpy sqlite3**

  2. For capturing Faces, Training, Detecting and taking Attendence log
      **python SmartFaceTrack**
     
**Advantages** <br>
  Fully automated attendance system <br>
  Reduces manual effort and human errors <br>
  Ensures accuracy and prevents duplicate attendance <br>
  Can be integrated into schools, offices, and organizations

**Future Enhancements** <br>
  Upgrade to deep learning models for improved accuracy <br>
  Integrate with cloud databases <br>
  Implement multi-face detection & recognition in one frame <br>

**Connect With Me** <br>
  LinkedIn Profile: https://www.linkedin.com/in/sai-teja-kotipalli-8b83b7259 <br>
  Email: kotipallisaiteja@gmail.com
