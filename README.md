## Quantum Mechanics for Machine Learning and Neural Network Math Base for Lower Limb Exoskeleton

### Objective

- **Quantum Mechanics for Machine Learning:**
  This topic focuses on understanding the fundamentals of `quantum computing`, including - how `qubits` are represented and manipulated using mathematical foundations such as vectors and probability theory.

  => The goal is to build a simple project that demonstrates how `quantum mechanics` concepts can be applied in `machine learning` without relying on `built-in functions`, helping to develop an intuitive understanding of how quantum systems work.

  - In this Quantum Mechanics, we will get use with `Kronecker product` whereas:
    ![alt text](pic/kronecker_product_example.png) ### Why Quantum Computing Uses Kronecker Product Instead of Normal Matrix Multiplication

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

        This operation creates a new vector that represents the full joint quantum system.

        ---

        ## Dimension Growth

        | Number of Qubits | State Dimension |
        |------------------|-----------------|
        | 1 | 2 |
        | 2 | 4 |
        | 3 | 8 |
        | n | \(2^n\) |

        This exponential growth is one of the reasons quantum computing is powerful.

        ---

        ## Example 1 — Combining Basis States

        Single qubit state:

        ```math
        |0\rangle =
        \begin{bmatrix}
        1\\
        0
        \end{bmatrix}
        ```


        Two qubits:

        ```math
        |00\rangle = |0\rangle \otimes |0\rangle
        ```

        MATLAB example:

        ```matlab
        kron([1;0],[1;0])

- **Neural Network in Lower Limb Exoskeleton:**
  This topic focuses on understanding `mathbase` of neural network foundation and it varitations including `type 2` and `fuzzy` for the Lower Limb Exoskeleton dataset.

  => The goal is to build a predicted action from the input of users to guess what is the next action (the subjects we focus on both `male` and `female` - no age limit)

### **Rework-Some Product**:

**Short about this**:

I realize my work will not be approved if I dont have any products or real things apply that can see/bring to users that will show my research reliable :(

So, I come up with this project.

> After develop the `Gesture Training` for self-build MLP Neural Network Model - I would like to use it to perform a `REAL MVP` about **Digital Twin with Glove-Sensors**

This is an example about the way it would work:

**1. Every first development geometry about the Project-Plane and Estimate the Palm-Plane**
![alt text](pic/sample-mvp.png)

**2. Calculated the Angle between Vectors in Project-Plane and Palm-Plane**
![alt text](pic/sample-mvp-v2.png)

**3. Start adding-in virtual sensors generators by calculating physics/math equation**
![alt text](pic/virtual_sensors.png)

**4. Converting vector into feature that input for MATLAB model**
![alt text](pic/feature_vector.png)

**5. Training process with MATLAB self-built model**
![alt text](pic/training_process_model.png)

**6. Predict based on Digital Twin**
![alt text](pic/predictive.png)

> This is still not as my imagination, i mean the Digital Twins part, the model is reflected quite well, but when want to improve we need more efficient rather than 90%
