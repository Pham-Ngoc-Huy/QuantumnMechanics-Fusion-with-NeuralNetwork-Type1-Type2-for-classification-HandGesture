# Day 4

**Question Raised:**

1. Input của neural network là gì ?
2. Output của neural network là gì ?
3. Tại sao paper cần neural network nếu đã có kinetostatic model ?
4. 512 gestures dùng để làm gì ?
5. Project của mình thày 4 strain gauges bằng Mediapipe như thế nào ?

## Review:

**Cách 1: Model-Based**

```math

\text{Sensor: `strain gauge voltage`} \\

\downarrow \\

\text{Calibration: local beam deformation} \space (\phi_{1} .. \phi_{4}) \\

\downarrow \\

\text{kinetostatic model:} \space (\theta / \text{finger configuration}) \\

\downarrow \\

\text{inverse kinematic:} \space \text{finger joint angles} \space (\alpha_1, \alpha_2, \alpha_3)
```

**Cách 2: Data-Driven**

Thay vì mỗi lần giải lại

> nonlinear equations $\rightarrow$ Newton-Raphson $\rightarrow$ solution

paper dùng:

> Neural network

để học mapping trực tiếp:

```math
\boxed{
    \text{sensor measurement}
}

\rightarrow

\boxed{
    \text{finger-joint variables}
}

```

> [!Note]
>
> Paper nói chính xác lý do là `Newton–Raphson` có `computational burden` và cần `iterative searching`, nên neural network được dùng để giảm `computational burden`
