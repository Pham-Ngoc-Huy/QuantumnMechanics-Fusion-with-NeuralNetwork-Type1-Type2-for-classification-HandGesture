# Skill: Docker Compose Setup & Deployment

## Description

This skill provides complete instructions, operational procedures, and planning for building, running, and maintaining the Docker & Docker Compose containerized environment for the **Quantum Mechanics Fusion with Neural Network (Type-1 & Type-2) Hand Gesture Classification** project.

---

## Architecture & Configuration

### 1. Dockerfile Specification

The `Dockerfile` builds a Python 3.12 environment tailored for computer vision (OpenCV, MediaPipe) and quantum/neural network data processing:

- **Base Image**: `python:3.12-slim`
- **System Dependencies**:
  - OpenCV runtime support: `libgl1`, `libglib2.0-0`, `libsm6`, `libxext6`, `libxrender-dev`, `libgomp1`
  - Video hardware tools: `v4l-utils`
  - Build essentials: `build-essential`
- **Application Directory**: `/app`
- **Python Package Installation**: Built using `requirements.txt` with cached wheel cleanup (`--no-cache-dir`).

### 2. Docker Compose Architecture

The `docker-compose.yml` configures the primary service (`hand-gesture-app`):

- **Build Context**: Root directory using `Dockerfile`.
- **Environment**:
  - `DISPLAY=${DISPLAY}`: Enables X11 GUI forwarding for OpenCV windows.
  - `QT_X11_NO_MITSHM=1`: Fixes potential shared memory issues in Qt/OpenCV windows inside containers.
  - `PYTHONUNBUFFERED=1`: Ensures real-time logging output.
- **Hardware Pass-through**:
  - `/dev/video0:/dev/video0`: Passes host webcam directly into the container.
- **Volumes**:
  - `/tmp/.X11-unix:/tmp/.X11-unix:rw`: Socket mount for X11 rendering.
  - `.:/app`: Live workspace mount for development.

---

## Step-by-Step Operations

### Step 1: Host Display Permissions

Before launching the container, grant X server access to local container clients on the host:

```bash
xhost +local:root
```

> **Note for macOS**: Ensure [XQuartz](https://www.xquartz.org/) is installed and running with _"Allow connections from network clients"_ checked in Preferences.

### Step 2: Build and Launch

Build the container image and launch the service:

```bash
docker compose up --build
```

To launch in background (detached mode):

```bash
docker compose up -d
```

### Step 3: Monitor Logs & Interactivity

View live container logs:

```bash
docker compose logs -f
```

Execute an interactive shell inside the running container:

```bash
docker compose exec hand-gesture-app bash
```

### Step 4: Stop Services

Gracefully stop and clean up containers:

```bash
docker compose down
```

---

## Troubleshooting & Maintenance Plan

| Category          | Issue                                                        | Resolution                                                                                                                                        |
| ----------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Camera Access** | `Cannot open camera /dev/video0`                             | Verify camera node via `ls /dev/video*`. Update the `devices` mapping in `docker-compose.yml` if camera index is different (e.g., `/dev/video1`). |
| **X11 / GUI**     | `qt.qpa.plugin: Could not load the Qt platform plugin "xcb"` | Run `xhost +local:root` on host before running `docker compose up`. Verify `DISPLAY` environment variable is set.                                 |
| **Dependencies**  | Package updates in `pyproject.toml` / `requirements.txt`     | Trigger clean rebuild: `docker compose build --no-cache`.                                                                                         |
