# Vision-Based Digital Twin of a Sensor Glove for Real-Time Hand Gesture Recognition
> This is continue develop to get the MVP for utilizing again the MLP that I have explored

```mermaid
flowchart LR
    A[Webcam] --> B[Hand Landmark Detector]
    B --> C[Virtual Sensor Generator<br/>Joint Angle Reconstruction]

    C --> D[Virtual Sensor Values<br/>22 Channels]
    C --> E[Digital Twin Viewer<br/>Animated Hand]

    D --> F[MATLAB Custom MLP]
    E --> F

    F --> G[Gesture Classification]
```

## 1. WebCam
## 2. Hand-Landmark Detector