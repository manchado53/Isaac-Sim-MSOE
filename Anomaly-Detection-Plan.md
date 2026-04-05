# Self-Supervised VLM Regression for Autonomous Door Inspection

This framework outlines the transition from a generic zero-shot Vision-Language Model (VLM) approach to a self-supervised, auto-labeled fine-tuning strategy. By leveraging the internal physics state of the **NVIDIA Isaac Sim digital twin**, we can train the **Qwen2-VL-2B model** to perform precise numerical state estimation of environmental features like doors.

---

## 1. The Core Pivot: From Description to Regression
While a zero-shot approach provides high-level scene awareness, it often lacks the precision required for security-grade anomaly detection. By pivoting to a self-supervised regression model, we use the simulator itself as the "teacher," allowing the robot to learn the visual difference between a fully closed door and one that is slightly ajar.

### Logic Breakdown
In Isaac Sim, doors are configured as articulated rigid bodies with revolute joints. This allows us to programmatically query the exact physical state of the asset.

* **The Label:** We use the `get_joint_positions()` function to retrieve the door asset's angle.
    * A joint position of **0.0 radians** indicates the door is flush with the frame (closed).
    * An angle of **1.57 radians** (90°) indicates it is fully open.
    * This value is normalized to a $0$–$1$ scale for training consistency.
* **The Prompt:** The VLM is fed an RGB image from the **Nova Carter's** front-mounted camera with a specific instruction:
    > "On a scale of 0 to 1, where 0 is closed and 1 is fully open, what is the state of the door in this image?"
* **The Training:** The system calculates a measurable error (loss) between the VLM's prediction and the synthetic ground truth. If the robot "sees" a door and the VLM predicts $0.8$, but the simulator’s physics data says $0.1$ (nearly closed), we have a loss value used to fine-tune the weights of Qwen2-VL.

---

## 2. Why VLM Over a Standard CNN?
While a Convolutional Neural Network (CNN) is effective for simple image classification, it falls short in a complex, multi-functional security environment.

* **Semantic Reasoning:** A CNN can tell you a door is open, but a VLM can understand the implication. A VLM can reason that an open door is normal at 2:00 PM but an anomaly at 2:00 AM.
* **Explainability:** When the system flags an anomaly, the VLM can provide a natural language description to a human operator (e.g., "The lab door is propped open by a trash can"), whereas a CNN provides only a numerical probability.
* **Multi-Tasking:** The same VLM used for door regression can simultaneously detect other anomalies like smoke, unauthorized personnel, or obstructions in fire exits without requiring a separate model for each task.

---

## 3. Solving the "No-Door" Problem
Since the Nova Carter camera captures data throughout its patrol, many frames will contain no doors. Forcing a regression output ($0$–$1$) on a plain wall would lead to "hallucinations."

### The Navigation Graph Solution
We utilize the navigation graph extracted from the **BIM data** to manage perception:

1.  **Spatial Triggering:** The robot only activates the "Door Inspection" prompt when it is within a specific radius of a doorway threshold node in the navigation graph.
2.  **Two-Step Prompting:** The model is fine-tuned to first answer a binary question:
    * **Step 1:** "Is a door visible in this frame?"
    * **Step 2:** If "Yes," perform the $0$–$1$ regression. If "No," output "None."

---

## 4. Implementation Workflow

### Data Collection
The collection process is fully automated within the Isaac Sim environment:
* The robot performs random patrols while the script captures RGB frames.
* Each frame is saved with its metadata: `[Image_ID, Door_Visible (True/False), Joint_Angle (0-1)]`.
* This generates thousands of labeled training points without manual human intervention.

### Fine-Tuning with LoRA
Because Qwen2-VL is a large model, we use **Low-Rank Adaptation (LoRA)**. This allows us to fine-tune only a small subset of the model's weights, making it an expert in reading door apertures while preserving its general language understanding.

### Evaluation Metrics
To measure the success of the pivot, we track:
* **F1-Score:** To evaluate how accurately the model identifies the presence of a door.
* **Mean Absolute Error ($MAE$):** To evaluate the precision of the angle estimation:
$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$