# AI-Driven Early Detection of Autism in Toddlers Using Multimodal Data

Autism Spectrum Disorder (ASD) is a neurodevelopmental condition that affects communication, behavior, and social interaction. Early diagnosis significantly improves intervention outcomes. This project presents a proof-of-concept AI solution that leverages non-invasive, video-based multimodal data to detect early behavioral signs of autism in toddlers.

## Objective

To build an AI-powered pipeline that uses camera-captured visual cues to detect early signs of autism in toddlers, enabling early intervention and diagnosis.

## Key Features Implemented

### 1. Eye Contact Analysis
- **Purpose**: Detect reduced or absent eye contact.
- **Approach**: Frame-by-frame analysis of toddler’s gaze direction using computer vision techniques.
- **Flag Triggered When**: Eye contact is below a defined threshold (e.g., less than 50%).

### 2. Repetitive Movement Detection
- **Purpose**: Identify repetitive behaviors such as hand flapping or body rocking.
- **Approach**: Motion pattern detection using pose estimation and/or a trained model.
- **Flag Triggered When**: Similar motion pattern is detected repeatedly (e.g., three or more repetitions).

### 3. Social Reciprocity Analysis
- **Purpose**: Assess social engagement through gestures or responses to interactions.
- **Approach**: Analyze presence of gestures or social behavior cues via body landmarks and hand movements.
- **Flag Triggered When**: Limited or no gestures or social interactions are observed.

## Tech Stack

- **Programming Language**: Python 3.8+
- **Frameworks and Libraries**:
  - **OpenCV** – For video processing and frame extraction
  - **NumPy** – Numerical operations and array handling
  - **TensorFlow/Keras** – For loading and using the trained model to detect repetitive behaviors
  - **Mediapipe** (optional) – For body/pose/hand landmark detection in gesture and social reciprocity analysis
  - **Matplotlib** – For visual debugging in notebooks (optional)
- **Development Tools**:
  - Google Collab Notebook – For prototyping and visualization
  - VS Code / Any Python IDE – For script development and debugging
- **Input Data Format**:
  - MP4 video files:
    - `eye_contact1.mp4` (gaze analysis)
    - `gesture_delay1.mp4` (repetitive behavior analysis)
    - `social.mp4` (social interaction cues)


## Project Structure

```
autism_detection/
├── models/
│   ├── eye_contact.py
│   ├── repetitive.py
│   ├── repetitive_gesture_model.h5
│   └── social_reciprocity.py
├── sample_data/
│   ├── eye_contact1.mp4
│   ├── gesture_delay1.mp4
│   └── social.mp4
├── .gitignore
├── LICENSE
├── README.md
├── autism_detector.py
├── gaze_detection.ipynb
├── repetitive_gesture.ipynb
├── social_reciprocratory.ipynb
└── requirements.txt
```

## How It Works

1. Each video is processed using specialized scripts in the `models/` directory:
   - `eye_contact.py` analyzes gaze direction and frame-wise face detection to compute the **eye contact percentage**.
   - `repetitive.py` uses a trained deep learning model (`repetitive_gesture_model.h5`) to detect **repetitive movements** and output a **confidence score**.
   - `social_reciprocity.py` evaluates the child’s gestures and reactions to identify **reduced social engagement**, yielding a **social reciprocity score**.

2. Each module returns a **quantitative score (0–100%)**, representing the severity or presence of the behavior.

3. The main script (`autism_detector.py`) aggregates the individual results into a structured report:
   ```python
   {
     "eye_contact_score": 42.67,
     "repetitive_behavior_score": 15.23,
     "social_reciprocity_score": 36.9
   }
   ```

4. Thresholds are applied internally to interpret the scores into flags (e.g., `eye_contact_score < 50` implies low eye contact), which can be used to make early-stage predictions or for visual dashboards.

5. The model is designed to be a **non-invasive, camera-based early screening tool**, making it practical for home or clinical use.
   

## Usage

1. Place your video files in the `sample_data/` directory with appropriate names:
   - `eye_contact1.mp4`
   - `gesture_delay1.mp4`
   - `social.mp4`

2. Run the main script:
   ```bash
   python autism_detector.py
   ```

3. Review the printed output for flags indicating the presence or absence of autism-related behaviors.

## Problem Statement Coverage

This solution addresses the following components from the problem statement:

- **Feature Identification**: Reduced eye contact, repetitive motor movements, and lack of social reciprocity are analyzed.
- **Eye Contact Analysis**: Using gaze direction estimation.
- **Repetitive Behavior Detection**: Based on pose estimation and/or trained model patterns.
- **Social Reciprocity Assessment**: Evaluated using body movement and gesture detection.

## Limitations and Future Work

- Current system is a proof of concept; performance depends on video quality and lighting.
- Uses basic threshold-based logic; future versions could integrate real-time deep learning-based detection.
- Adding audio analysis and larger annotated datasets would improve reliability and coverage.

## Conclusion

This project demonstrates a modular and lightweight AI pipeline for early autism detection using multimodal video data. It offers an accessible and non-invasive approach to aid clinicians and caregivers in early screening and diagnosis of Autism Spectrum Disorder.
