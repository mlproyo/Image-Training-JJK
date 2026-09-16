# Domain Expansion: Hand Gesture Recognition 

A real-time computer vision project that detects hand gestures via webcam and triggers Jujutsu Kaisen **Domain Expansion** effects — matching hand signs to Gojo's *Infinite Void*, Sukuna's *Malevolent Shrine*, and *Mahoraga* (this one is called Idle Death Gamble in the code, the Hakari expansion didn't work but I kept the name and switched to Mahoraga) — complete with sound effects and screen tint overlays.

Built with **OpenCV**, **cvzone**, and a **Teachable Machine**–trained Keras classifier.

##  Features

- Real-time hand tracking with a drawn skeleton overlay for up to 2 hands
- Two separate classifiers:
  - **Single-hand model** → detects Gojo's *Infinite Void*
  - **Two-hand model** → detects Sukuna's *Malevolent Shrine* and *Mahoraga*
- Gesture confirmation delay (2 seconds held) to avoid false positives
- Full-screen color tint effect once a domain is confirmed
- Matching audio playback for each confirmed domain

##  Demo

*(Add a GIF or screen recording here showing the gestures and effects in action)*

##  Requirements

- Python 3.x
- [OpenCV](https://pypi.org/project/opencv-python/)
- [cvzone](https://pypi.org/project/cvzone/)
- numpy
- pygame

All dependencies are listed in `requirements.txt`.

##  Setup & Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv handenv
   ```

3. **Activate the virtual environment**
   ```bash
   handenv\Scripts\activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

##  Usage

Run the main script:

```bash
python test.py
```

- Hold up **one hand** in the correct sign to trigger *Infinite Void*.
- Hold up **two hands** in the correct sign to trigger *Malevolent Shrine* or *Mahoraga*.
- Hold the gesture steady for **2 seconds** to confirm the domain.
- Press **`q`** to quit the application.

##  Project Structure

```
├── test.py                # Main application script
├── requirements.txt       # Python dependencies
├── Model/                 # Single-hand classifier
│   ├── keras_model.h5
│   └── labels.txt
├── Model2/                # Two-hand classifier
│   ├── keras_model.h5
│   └── labels.txt
└── Audio/                 # Domain expansion sound effects
    ├── Gojo Domain.mp3
    ├── Sukuna Domain.mp3
    └── Mahoraga Audio.mp3
```

##  How It Works

1. **Hand detection** — `cvzone`'s `HandDetector` locates up to two hands per frame and returns their landmarks.
2. **Preprocessing** — The detected hand region is cropped, centered, and resized onto a fixed 300x300 white canvas (`imgWhite`) to normalize input for the classifier.
3. **Classification** — The prepared image is passed to the relevant Keras model (`Model/` for one hand, `Model2/` for two hands) to predict the gesture label.
4. **Confirmation** — A gesture must be held consistently for **2 seconds** before it's considered "confirmed," reducing false triggers.
5. **Effects** — Once confirmed, matching audio plays and, after a short delay, a colored tint is overlaid on the video feed to simulate the domain's visual effect.

##  Notes

- The *Mahoraga* gesture was originally trained as "Idle Death Gamble," but was renamed since no matching audio existed for that domain at the time this project was built.
- Detection and tracking confidence thresholds are tuned relatively low (`0.3`) for smoother real-time performance — feel free to adjust in `HandDetector(...)` for your setup.

##  Credits
- Hand tracking powered by [cvzone](https://github.com/cvzone/cvzone)
- Gesture classification trained with [Teachable Machine](https://teachablemachine.withgoogle.com/)
- Inspired by *Jujutsu Kaisen*
