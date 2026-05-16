# 🐾 Animal Detection & Alert System

Real-time dangerous animal detection using **YOLOv8** with a **Streamlit web interface**.  
Classifies detected animals into danger levels — HIGH / MEDIUM / LOW — and raises visual alerts.

---

## 🚀 Features

- **YOLOv8-powered detection** — nano / small / medium model support
- **3 input modes** — Image upload, Video upload, Live webcam
- **Danger level classification** with color-coded alerts (🔴 HIGH / 🟡 MEDIUM / 🟢 LOW)
- **Hardware integration** (optional) — Arduino LED/buzzer + GPS + Twilio SMS alerts
- **Configurable confidence threshold** via sidebar

---

## 🐉 Danger Level Map

| Animal | Danger Level |
|--------|-------------|
| Lion, Tiger, Bear, Elephant | 🔴 HIGH |
| Cow, Horse, Zebra | 🟡 MEDIUM |
| Dog, Cat, Sheep, Bird | 🟢 LOW |

---

## 🛠️ Installation

```bash
pip install -r requirements.txt
```

## ▶️ Run Web App

```bash
streamlit run app.py
```

Open: [http://localhost:8501](http://localhost:8501)

---

## 📁 Project Structure

```
Animal_Detection_Alert_System/
├── app.py              # Streamlit web UI
├── detection.py        # Core YOLOv8 detection logic
├── requirements.txt
├── README.md
└── sample_images/      # Test images
```

---

## 🔧 Hardware Mode (Optional)

The original system supports:
- **Arduino** — LED (green/orange/red) + buzzer alerts via serial
- **NEO-6M GPS module** — real-time location tracking
- **Twilio SMS** — alert messages with Google Maps link

To enable, configure `detection_hardware.py` with your credentials.

---

## 📸 Tech Stack

- [Ultralytics YOLOv8](https://docs.ultralytics.com)
- [Streamlit](https://streamlit.io)
- [OpenCV](https://opencv.org)
- Python 3.8+