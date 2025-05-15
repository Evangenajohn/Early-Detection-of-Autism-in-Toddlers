from models.eye_contact import analyze_eye_contact
from models.repetitive import predict_repetitive_movement
from models.social_reciprocity import analyze_social_reciprocity

def autism_prediction(gaze_video, gesture_video, social_video):
    eye_contact = analyze_eye_contact(gaze_video)
    repetitive_score = predict_repetitive_movement(gesture_video)
    social_results = analyze_social_reciprocity(social_video)

    # Example thresholds (tune as needed)
    print("Eye contact result: ", eye_contact)  # 👈 Add this line
    eye_contact_flag = eye_contact < 50  # original line

    repetitive_flag = repetitive_score > 0.5  # high repetitive gestures
    social_flag = social_results['social_response_score'] < 0.2  # low social reciprocity

    autism_likelihood = sum([eye_contact_flag, repetitive_flag, social_flag]) / 3

    result = {
        "eye_contact_percent": eye_contact,
        "repetitive_movement_prob": repetitive_score,
        "social_response_score": social_results['social_response_score'],
        "social_reciprocity_level": social_results['social_reciprocity_level'],
        "autism_likelihood_score": autism_likelihood
    }
    return result

if __name__ == "__main__":
    gaze_video = "sample_data\eye_contact1.mp4"
    gesture_video = "sample_data\gesture_delay1.mp4"
    social_video = "sample_data\social.mp4"

    results = autism_prediction(gaze_video, gesture_video, social_video)
    print("Autism Prediction Results:")
    print(results)
