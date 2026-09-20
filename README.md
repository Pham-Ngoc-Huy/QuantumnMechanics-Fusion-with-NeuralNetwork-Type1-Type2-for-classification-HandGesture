# System Identification with Intepreting Quantum Mechanics for Machine Learning, Neural Network, and Deep Learning:

## 1. Quantum Mechanics for Machine Learning and Neural Network Math Base for Lower Limb Exoskeleton

### Objective

- **Quantum Mechanics for Machine Learning:**
  This topic focuses on understanding the fundamentals of `quantum computing`, including how `qubits` are represented and manipulated using mathematical foundations such as vectors and probability theory.

=> The goal is to build a simple `Variational Quantum Classifier (VQC)` project that demonstrates how `quantum mechanics` concepts can be applied in `machine learning` — implemented from scratch in MATLAB, without any built-in quantum toolbox, helping to develop an intuitive understanding of how quantum systems work.

### Why Quantum Computing Uses the Kronecker Product

Quantum computation uses **qubits** instead of classical bits. Unlike classical bits that can only be `0` or `1`, a qubit can exist in a **superposition** of both states:

```math
|\psi\rangle = \alpha |0\rangle + \beta |1\rangle
```

where:

- $\alpha, \beta \in \mathbb{C}$ are complex probability amplitudes
- $|\alpha|^2 + |\beta|^2 = 1$

This means even a single qubit exists in a **2-dimensional vector space**.

---

### Why Normal Matrix Multiplication Is Not Enough

When we combine multiple qubits, we must represent **all possible combinations of their states**.

For example, a 2-qubit system must represent:

```math
|00\rangle,\ |01\rangle,\ |10\rangle,\ |11\rangle
```

This corresponds to a **4-dimensional space**.

Normal matrix multiplication only transforms vectors **within the same space** and cannot increase dimensionality. However, quantum systems require the space to grow **exponentially** when qubits are combined.

---

### Tensor (Kronecker) Product

To correctly combine quantum systems, we use the **Kronecker product**, also known as the tensor product.

```math
|\psi_{total}\rangle = |\psi_1\rangle \otimes |\psi_2\rangle
```

This operation creates a new vector that represents the full joint quantum system. In MATLAB:

```matlab
kron([1;0],[1;0])   % (2 x 1) tensor (2 x 1) = (4 x 1)  ->  |00>
```

![alt text](pic/kronecker_product_example.png)

---

### Dimension Growth

| Number of Qubits | State Dimension |
| ---------------- | --------------- |
| 1                | 2               |
| 2                | 4               |
| 3                | 8               |
| n                | \(2^n\)         |

This exponential growth is one of the reasons quantum computing is powerful — and it is exactly why the classifier below builds its gates and states with the Kronecker product.

### 1.1 Dataset -- `quantum/data_qc.csv`

A small, self-given classification dataset:

| x1  | x2  | label |
| --- | --- | ----- |
| 0.1 | 0.2 | 1     |
| 0.2 | 0.1 | 1     |
| 2.8 | 3.0 | -1    |
| 3.0 | 2.7 | -1    |

Two features ($x_1$, $x_2$) → **2 qubits, one per feature**. The label is the output, not a feature.

### 1.2 Training -- `quantum/qvc_quantum_training.m`

A from-scratch **Variational Quantum Classifier (VQC)** pipeline:

1. **Normalize** each feature to the angle range `[0, π]`: `x / max_val * π`.
2. **Initial state**: `|00> = [1; 0; 0; 0]`.
3. **Encoding circuit (angle encoding)**: a rotation gate about the $Y$-axis,

   ```math
   RY(\theta) =
   \begin{bmatrix}
   \cos(\theta/2) & -\sin(\theta/2)\\
   \sin(\theta/2) & \cos(\theta/2)
   \end{bmatrix}
   ```

   applied on each qubit and combined with the Kronecker product: `U = RY(x1) ⊗ RY(x2)`.

4. **Variational circuit**: a second, trainable rotation `W = RY(θ1) ⊗ RY(θ2)`.
5. **Measurement**: expectation value of the observable `O = Z ⊗ I`:

   ```math
   f = \langle \psi | O | \psi \rangle \in [-1, 1]
   ```

   with `Z = [1 0; 0 -1]` (Pauli-Z) and `I` the identity — only the real part is kept.

6. **Loss**: mean-square error `(f - y)^2` between the quantum output and the `{±1}` label.
7. **Gradient — parameter-shift rule** (quantum's replacement for backprop): the circuit is evaluated twice with shifted angles,

   ```math
   \frac{\partial f}{\partial \theta} = \frac{f(\theta + \pi/2) - f(\theta - \pi/2)}{2}
   ```

   then updated as `theta = theta - eta * grad`.

8. **Hyperparameters**: `eta = 0.1`, initial `θ1 = 0.5`, `θ2 = -0.3`, 10 epochs.

The trained parameters and normalization metadata are stored in `vqc_model.mat`.

Output: training-loss curve and the model-output curve (the output should squeeze toward the target labels `{-1, +1}`).

### 1.3 Testing -- `quantum/qvc_quantum_testing.m`

- Reloads `vqc_model.mat` and re-applies the same normalization and angle encoding.
- Recomputes `f = Re(⟨ψ|O|ψ⟩)` and classifies:

```matlab
if f >= 0
    pred = 1;
else
    pred = -1;
end
```

- Reports the accuracy on the dataset and plots predicted vs. true labels.

## 2. Self-Built Library -- Math & Machine-Learning Methodology (`library/`)

### Objective

- The linear-algebra and machine-learning **methodology** is implemented from scratch in pure Python (`library/`), with no `numpy` or any math library — every matrix operation used later by the quantum classifier (section 1), the neural networks (section 3) and the digital-twin runtime (section 4) is first understood and hand-written here.

=> Classical math-building blocks written as clean Python functions, re-exported from `library/__init__.py`.

### 2.1 Basic matrix & vector operations

| File                 | Function                       | What it does                                             |
| -------------------- | ------------------------------ | -------------------------------------------------------- |
| `matrix_multiply.py` | `matrix_multiply(A, B)`        | classic row × column matrix multiplication               |
| `transpose.py`       | `transpose(M)`                 | swap rows ↔ columns (`Mᵀ`)                              |
| `scalar_multiply.py` | `scalar_multiply(a, b)`        | dot product `a·b` of two vectors                         |
|                      | `projection(a, b)`             | orthogonal projection of `a` onto `b`: `(a·b / b·b) · b` |
| `operator.py`        | `add(a, b)` / `subtract(a, b)` | element-wise vector addition / subtraction               |
| `identity_matrix.py` | `identity_matrix(M)`           | builds the identity matrix `I` for a square matrix       |

### 2.2 Determinant — `determinant.py`

Recursive **cofactor (Laplace) expansion** along the first row:

```math
\det(A) = \sum_{j=1}^{n} (-1)^{1+j} a_{1j} \, M_{1j}
```

where `M_{1j}` is the minor obtained by deleting row 1 and column `j` (`get_minor`).

### 2.3 QR decomposition -- `QR_decomposition.py`

**Gram–Schmidt** orthogonalization: produces `Q` (orthonormal columns) and upper-triangular `R` (the projection coefficients) such that `A = QR`:


```math
q_k = \frac{a_k - \sum_{i \lt k} (a_k \cdot q_i)\, q_i}{\| a_k - \sum_{i \lt k} (a_k \cdot q_i)\, q_i \|}
```

The normalization step is the Euclidean norm; each orthogonal basis vector is built by subtracting the projections (`projection`) of `a_k` onto the already-computed `q_i`.

### 2.4 QR algorithm for eigenvalues -- `QR_algorithm.py`

Iterative **QR iteration**: repeat `A = QR` then `A ← R·Q`. Over iterations the matrix converges so the **eigenvalues appear on the diagonal** (default `max_iter = 100`):

```python
while iter < max_iter:
    Q, R = QR_decomposition(A)
    A = matrix_multiply(R, transpose(Q))
```

This is the classical linear-algebra foundation (matrix factorization / spectral decomposition) behind concepts like principal components and expectation values that the other parts rely on conceptually.

## 3. Neural Network in Lower Limb Exoskeleton:

### Objective

- **Math base of neural network foundation** and its variations (`MLP`, `Fuzzy Neural Network`) — all self-implemented in MATLAB (no deep-learning toolbox), running on the **Lower Limb Exoskeleton** dataset.

=> The goal is to build a predictor that guesses the user's next action (walking velocity class) from their inputs, covering both `male` and `female` subjects — no age limit.

### 3.1 Data -- `reading_data_velocity.m`

For every participant and every walking-velocity trial, the raw study contains per-limb EMG and normalized joint torque recordings. The script scans each participant's `Processed_Data.mat`, aligns the columns, parses the `Metadata.txt`, then merges everything into one matrix `all_data_full`, saved as `processed_data_velocity_classification.mat`:

| Group            | Channels | Columns                                                                         | Notes                |
| ---------------- | -------- | ------------------------------------------------------------------------------- | -------------------- |
| Labels           | 2        | participant, velocity                                                           | velocity → 7 classes |
| EMG right leg    | 8        | St1/St2 × {VL, BF, TA, GAL}                                                     | filtered EMG         |
| EMG left leg     | 4        | St1 × {VL, BF, TA, GAL}                                                         | filtered EMG         |
| Torque right leg | 24       | St1/St2 × {Pelvis, Hip, Knee, Ankle} × {X, Y, Z}                                | normalized torque    |
| Torque left leg  | 9        | St1 × {Hip, Knee, Ankle} × {X, Y, Z}                                            | normalized torque    |
| Metadata         | 6        | gender (1=Male, 2=Female), age, body_height, body_mass, leg_length, foot_length |                      |

=> Feature matrix `X` = 51 columns; target = velocity label (7 classes).

### 3.2 Self-built MLP -- `main_nn_mlp.m`

A from-scratch multilayer perceptron for velocity classification:

- **Preprocessing**: min-max normalize each feature to `[0,1]`; one-hot encode the 7 velocity classes.
- **Train/test split**: 70% / 30%, `rng(42)` + `randperm` (fixed seed for reproducibility).
- **Architecture**: 51 → 512 → 256 → 128 → 64 → 7 (4 hidden layers).
- **Activation**: bipolar sigmoid (`tanh`) on hidden layers, `softmax` on output.
- **Loss**: categorical cross-entropy (clipped at `1e-12`).
- **Optimizer**: mini-batch SGD with **Adam** — `eta = 0.001`, `beta1 = 0.9`, `beta2 = 0.999`, `eps = 1e-8`, `batch_size = 256`, `maxepoch = 200`.
- **Initialization**: Xavier/uniform — `limit = sqrt(6 / (n_in + n_out))`; a bias row of ones is prepended to every layer's input.

Both forward (`feedforward_multihidden.m`) and backprop (`backprop_multihidden.m` — where the softmax + cross-entropy gradient collapses to `ŷ − t` and the tanh derivative is `0.5(1 − o²)`) are hand-written.

Output: train/test loss & accuracy curves, plus confusion matrices for train and test.

### 3.3 Fuzzy Neural Network (TSK-style) -- `main_fuzzy_nn_mlp.m`

Same dataset, but the 51 features first pass through a **fuzzy layer** before the MLP:

- **Layer 1 — Fuzzification**: `K = 3` Gaussian membership functions (Low/Medium/High) per feature → `51 × 3 = 153` rules.
- **Layers 2–3 — Firing & normalization**: rule firing degree `W_fire`, normalized by column sum → `W_norm`.
- **Layer 4 — Consequents**: Takagi–Sugeno linear rules `F = P·x + Q` (153 × 7), weighted by `W_norm`.
- **Layer 5 — Defuzzification**: weighted sum over all rules → `7 × batch` crisp vector.
- **MLP head**: 7 → 64 → 32 → 7 (`ReLU` hidden, chosen for compatibility with the `[0,1]` fuzzy outputs, `softmax` output).

The whole system (MF centers `C`, widths `Σ`, consequents `P`/`Q`, plus MLP weights) is trained **end-to-end** with Adam — `eta = 0.001`, `maxepoch = 300`, `batch_size = 512`. `fuzzy_backprop.m` propagates `ŷ − t` back through the softmax, the defuzzification layer, and all the way to the membership-function gradients; `Σ` is clamped to stay positive.

Output: training curves, confusion matrices, and the learned fuzzy membership functions per muscle.

### 3.4 Hand-gesture variant & math-base scripts

- `reading_data_human_activities.m` + `main_nn_mlp_human_activities.m`: the same pipeline applied to the `DB9` hand dataset — 19 finger-joint angle features, `[128, 64]` hidden, exports `gesture_model.mat` (reused by the Digital-Twin MVP later).
- `MLP_for_regression.m` / `feedforward_nn.m` / `GD_MLP.m`: single-hidden-layer MLP that learns a nonlinear dynamic (NARX-like) system by regression — the original math-base playground.
- `gradient_descent.m`, `bipolar_sigmoid.m`, `monopolar_sigmoid.m`: low-level building blocks (plain GD for a single neuron, tanh and logistic activations).

## 4. Rework-Some Product — Digital-Twin MVP:

**Short about this**:

Any research needs something real to show, so after building the `Gesture Training` (self-built MLP in section 3) I turned it into a `REAL MVP`: a **Digital Twin with Glove-Sensors** — MediaPipe replaces the physical sensor glove, and the MATLAB-trained MLP does the live classification in Python.

### 4.1 Codebase Tree

The product is split into two parts: the **MATLAB exploration** (`neural_network/`, `quantum/`, `library/`) and this **Python runtime** that consumes the trained model:

```
.
├── main.py                         # entry point — the app is triggered from here
├── Dockerfile                      # container image for the runtime
├── docker-compose.yml              # one-command run (webcam + X11 GUI passthrough)
├── config/
│   ├── config.py                   # YAML loader (Config)
│   └── config.yaml                 # model paths, landmarks, connections, sensors, gesture labels
├── camera/
│   └── webcam.py                   # Webcam — OpenCV capture, converts frame to MediaPipe Image
├── detector/
│   ├── base_detector.py            # abstract detector interface
│   ├── hand_landmarker.py          # HandLandmarkerDetector — MediaPipe LIVE_STREAM mode
│   └── callback.py                 # LandmarkVisualizer — wired as the detection callback
├── models/
│   ├── hand_landmarker.task        # MediaPipe pretrained hand-landmark model
│   └── gesture_model.mat           # MLP trained by main_nn_mlp_human_activities.m (section 3)
├── virtual_sensor/                 # the "virtual glove" — turns landmarks into angles/features
│   ├── generator.py                # GenerateVirtualSensor — computes every configured sensor angle
│   ├── vectorize.py                # MatlabFeatureVector — sensors → 19-dim feature vector
│   ├── classifier.py               # GestureClassifier — loads .mat, normalizes, MLP forward (softmax)
│   ├── geometry/
│   │   ├── estimate_palm_plane.py  # palm normal via cross product of wrist/index-MCP/little-MCP
│   │   ├── project_vector.py       # project a 3D vector onto the palm plane
│   │   └── angle_between.py        # angle between vectors by the cosine rule (degrees)
│   └── kinematics/
│       ├── flexion.py              # MCP/PIP/DIP flexion angles
│       ├── abduction.py            # angle between adjacent fingers (projected on palm plane)
│       └── wrist.py                # wrist flexion/abduction (disabled in config.yaml)
├── neural_network/                 # MATLAB: self-built MLP + fuzzy NN (section 3)
├── quantum/                        # MATLAB: VQC (section 1)
├── log/                            # shared logger
└── pic/                            # screenshots
```

The feature vector ordering follows `config.yaml -> sensors` (with `mcp2_a` dropped because that dataset column was NaN), matching the 19 features the MATLAB model was trained on in section 3.2.

### 4.2 How It Is Triggered — `main.py`

```python
config  = Config("config/config.yaml")
callback = LandmarkVisualizer(config=config)                     # draws + runs the whole pipeline

with HandLandmarkerDetector(model_path=config.model_path, callback=callback) as detector:
    with Webcam() as camera:                                     # opens /dev/video0
        while True:
            frame, mp_image = camera.read()
            detector.detect(image=mp_image, timestamp=timestamp) # MediaPipe async landmark detection
            callback.draw(frame=frame)                           # virtual sensors + prediction + drawing
            camera.show(frame=frame)                             # OpenCV window
            timestamp += int(1000 / config.fps)                  # ms per frame at config.fps
```

Per frame, `LandmarkVisualizer.draw()` (detector/callback.py) runs the **Digital-Twin pipeline**:

1. **Detect** — MediaPipe returns 21 hand landmarks (`wrist`, `thumb_cmc`, ..., `pinky_tip`) in `LIVE_STREAM` mode.
2. **Estimate the palm plane** — `EstimatePalmPlaneCross`: `palm_normal = (index_mcp - wrist) × (little_mcp - wrist)` normalized; `palm_center` = centroid of the three points.
3. **Virtual sensors** — `GenerateVirtualSensor` computes, per `config.yaml -> sensors`:
   - `flexion` (`FlexionAngle`): angle at the joint between `(parent→joint)` and `(joint→child)` vectors;
   - `abduction` (`AbductionAngle`): project two adjacent finger vectors onto the palm plane, then the angle between them.
4. **Feature vector** — `MatlabFeatureVector` concatenates the sensors in YAML order → `19`-dim vector.
5. **Predict** — `GestureClassifier` loads `gesture_model.mat`, min-max normalizes with the stored `X_min/X_max`, runs the same MLP forward pass (`bipolar_sigmoid` hidden layers + softmax), takes `argmax`, then maps the class to the human-readable activity via `config.yaml -> dimension_activities` (e.g. class 205 → "Abduction of all fingers").

Everything is overlaid live: hand skeleton, palm plane + normal, vector projection, included angle, virtual-sensor values, the MATLAB feature vector, and the `Gesture / Activity / Confidence` readout.

### 4.3 Running

**Locally** (need a webcam and the MediaPipe model in `models/`):

```bash
python main.py        # press 'q' to quit
```

**With Docker** — the container is built from `Dockerfile` (Python 3.12-slim + OpenCV/MediaPipe system libs) and `docker-compose.yml` wires the host webcam (`/dev/video0`) and the X11 socket through so the OpenCV window appears on your display (`DISPLAY`, `QT_X11_NO_MITSHM=1`, `- /tmp/.X11-unix`):

```bash
docker-compose up --build
```

Note: Docker is for Linux hosts with a working X11 (or XQuartz on macOS); the code volume is mounted at `/app` so edits are live.

### 4.4 Screenshots

![alt text](pic/sample-mvp.png)

![alt text](pic/sample-mvp-v2.png)

![alt text](pic/virtual_sensors.png)

![alt text](pic/feature_vector.png)

![alt text](pic/training_process_model.png)

![alt text](pic/predictive.png)

## Conclusion:

Even though it can classification some gestures right, but still as not expectations, so i will update this project to another stage (which mean another approach in solution [maybe not `classification` but `regression`])

> I will update new repository for this approach guys --> Stay tune :3
