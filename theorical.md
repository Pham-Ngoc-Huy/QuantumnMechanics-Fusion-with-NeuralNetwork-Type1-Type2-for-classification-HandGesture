# Vision-Based Digital Twin of a Sensor Glove for Real-Time Hand Gesture Recognition
> This is continue develop to get the MVP for utilizing again the MLP that I have explored

```bash
                   Webcam
                      │
                      ▼
            Hand Landmark Detector
                      │
                      ▼
        Virtual Sensor Generator
      (Joint Angle Reconstruction)
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
 Virtual Sensor Values        Digital Twin Viewer
 (22 Channels)               (Animated Hand)
        │                           │
        └─────────────┬─────────────┘
                      ▼
             MATLAB Custom MLP
                      │
                      ▼
             Gesture Classification
```

## 1. WebCam
## 2. Hand-Landmark Detector