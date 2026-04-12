# Related Work: Mobile Robots + VLMs for Building Inspection

Last updated: 2026-04-11

---

## Commercial Products

### Boston Dynamics Spot + Orbit 5.0 (Nov 2024)
- **What:** Quadruped patrol robot with VLM-style vision prompts for anomaly detection (debris, spills, corrosion, missing fire extinguishers)
- **Deployed at:** Cargill oilseed plants, POSCO steel facilities
- **Training data:** Real-world captured images, manually configured prompts per site
- **Limitation:** No synthetic training, no digital twin, no BIM automation. Requires manual per-site engineering for each new deployment.
- **Robot cost:** ~$75K+
- **Link:** https://bostondynamics.com/case-studies/spot-at-cargill/
- **Relevance:** Closest commercial competitor. Differentiator: they use real images + manual config; we use BIM → synthetic data → auto-generated training.

### NVIDIA Nova Carter + ReMEmbR (Nov 2024)
- **What:** Nova Carter driven around an NVIDIA office for ~25 min using VILA captioning VLM to build a spatial memory database. Robot answers natural language questions about the environment.
- **Limitation:** Research demo only. Not anomaly detection. No sim training. No deployment.
- **Link:** https://developer.nvidia.com/blog/using-generative-ai-to-enable-robots-to-reason-and-act-with-remembr/
- **Relevance:** Shows VLM-augmented spatial understanding on Nova Carter platform, but no anomaly detection or synthetic data pipeline.

### Knightscope (K5, K7)
- **What:** Autonomous security patrol robots deployed at NYPD, malls, corporate campuses. K7 (late 2025) adds off-road capability.
- **Limitation:** Traditional CV only (license plate readers, thermal cameras). Rule-based detection. No VLMs, no foundation models.
- **Link:** https://www.roboticstomorrow.com/news/2025/11/13/knightscope-unveils-the-all-new-k7-autonomous-security-robot/25804/
- **Relevance:** Shows market demand for patrol robots but no visual understanding capability.

### Cobalt Robotics
- **What:** Indoor security robot with human-in-the-loop escalation.
- **Limitation:** Conventional CV for person/object detection. No VLM integration.
- **Relevance:** Similar use case (indoor patrol) but no foundation model reasoning.

---

## Academic Papers

### InspectVLM: Unified in Theory, Unreliable in Practice (2025)
- **arXiv:** 2508.01921
- **What:** Unified Florence-2-based VLM for drone inspection of wind turbines, solar farms, rooftops. Tested on 292K images.
- **Key finding:** VLMs matched specialized models for classification but **failed badly at fine-grained object detection** — produced degenerate bounding boxes and exhibited "brittle language behavior" (collapsing to memorized responses by epoch 4).
- **Relevance:** Confirms our 0% generalization finding. Fine-tuned VLMs memorize appearance rather than learning concepts. This is the central open problem.

### VLM-based Anomaly Detection in Robotic Labs (2025)
- **arXiv:** 2506.05405
- **What:** GPT-4o + Qwen2.5-VL-72B tested on anomaly detection in a real robotic lab. GPT-4o reached 79.2% accuracy with hierarchical prompting; Qwen2.5-VL only 61.4%.
- **Key finding:** Off-the-shelf VLMs need heavy prompt engineering for domain-specific anomaly detection. Smaller models struggle significantly.
- **Relevance:** Shows that even large VLMs (72B) underperform on domain-specific visual anomaly tasks without fine-tuning. Motivates our fine-tuning approach.

### MonitorVLM: Safety Violation Detection in Mining (2025)
- **arXiv:** 2510.03666
- **What:** VLM framework for safety violation detection in mining operations.
- **Relevance:** Different domain (mining) but same pattern — VLM for visual safety monitoring in industrial environments.

### MolmoBot (Allen AI, 2025)
- **What:** VLM-based manipulation policy achieving zero-shot sim-to-real transfer without real-world fine-tuning.
- **Key finding:** Proves sim-to-real VLM transfer is viable — but for manipulation, not inspection.
- **Link:** https://allenai.org/blog/molmobot-robot-manipulation
- **Relevance:** Validates that VLMs can transfer from sim to real. Different task (manipulation vs inspection) but same paradigm.

### ReBot: Real-to-Sim-to-Real Synthetic Video Pipeline (2025)
- **What:** Synthetic video pipeline improving OpenVLA success rates by 21.8% in simulation and 20% in real world.
- **Key finding:** Synthetic data improves VLM/VLA real-world performance measurably.
- **Relevance:** Validates synthetic data → VLM pipeline. Manipulation focused, but proves the approach works.

### Language-as-Cost Hazard Mapping (2025)
- **arXiv:** 2508.03138
- **What:** Uses VLMs for proactive hazard avoidance during robot navigation via zero-shot language-based cost maps.
- **Relevance:** VLM-guided navigation, but zero-shot only (no fine-tuning, no synthetic data).

### Fully Automated Synthetic BIM Dataset Generation (2025)
- **Source:** ScienceDirect
- **What:** Automated pipeline for generating synthetic BIM datasets using ResNet + Revit API.
- **Limitation:** For floor plan generation, not VLM training data.
- **Relevance:** Only paper found using BIM for synthetic dataset generation — but not for VLMs or anomaly detection.

---

## Identified Gaps (What Has NOT Been Done)

1. **BIM → synthetic VLM training data:** Nobody has used architectural BIM data to auto-generate labeled training datasets for Vision-Language Models.

2. **Digital twin as VLM data factory for building inspection:** Spot uses real images. Academic papers use real images or generic sim environments. No one uses a building-specific digital twin derived from BIM to train a VLM.

3. **End-to-end pipeline (BIM → digital twin → synthetic anomalies → fine-tuned VLM → mobile robot):** The individual components exist but this integration has not been published.

4. **Sim-to-real transfer for VLM-based visual inspection:** Well-established for RL navigation policies and manipulation (MolmoBot), but **untested for VLM-based visual inspection from synthetic building environments.**

5. **Multi-anomaly generalization from synthetic data:** Current work handles single tasks (door state OR spill detection). A general-purpose building anomaly VLM trained on diverse synthetic scenarios does not exist.

6. **VLM generalization brittleness:** Both InspectVLM (2025) and our own results (0% on unseen doors) show that fine-tuned VLMs memorize appearance. Systematic study of how synthetic data diversity (domain randomization) addresses this is missing.

---

## Our Differentiators vs. Existing Work

| Dimension | Spot (Boston Dynamics) | Academic VLM Papers | Our System |
|---|---|---|---|
| Training data source | Real images | Real images | **Synthetic from BIM digital twin** |
| New building setup | Manual per-site config | N/A | **Import BIM file → auto-generate** |
| Labeling effort | Manual | Manual | **Zero — physics provides labels** |
| Anomaly generation | Must occur naturally | Staged or found | **Programmatic — simulate anything** |
| Robot platform | $75K+ Spot | Various | Nova Carter (~$5K) |
| Scalability | Per-site engineering | Per-dataset collection | **Pipeline: any BIM → trained VLM** |
