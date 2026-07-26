import cv2
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]
class LandmarkPrinter:
    def __call__(self, result, output_image, timestamp):
        if len(result.hand_landmarks) == 0:
            return
        print("="*80)
        print(f"Timestamp: {timestamp}")
        hand=result.hand_landmarks[0]
        for idx, lm in enumerate(hand):
            print(
                f"{idx:02d} "
                f"x={lm.x:.4f} "
                f"y={lm.y:.4f} "
                f"z={lm.z:.4f}"
            )

class LandmarkVisualizer:
    def __init__(self):
        self.result=None
    def __call__(self, result, output_image, timestamp):
        self.result=result
        if not result.hand_landmarks:
            return
        for idx, lm in enumerate(result.hand_landmarks[0]):
            print(
                f"{idx:02d} "
                f"x={lm.x:.4f} "
                f"y={lm.y:.4f} "
                f"z={lm.z:.4f}"
            )
        print("Callback")

    def draw(self, frame):
        if self.result is None:
            return

        for hand in self.result.hand_landmarks:
            self._draw_hand(frame, hand)

    def _draw_hand(self, frame, hand_landmarks):
        h, w = frame.shape[:2]
        
        points = []
    
        for lm in hand_landmarks:
    
            x = int(lm.x * w)
            y = int(lm.y * h)
    
            points.append((x, y))
    
            cv2.circle(frame, (x, y), 4, (0,255,0), -1)
    
        for start, end in HAND_CONNECTIONS:
    
            cv2.line(
                frame,
                points[start],
                points[end],
                (255,0,0),
                2
            )
        
