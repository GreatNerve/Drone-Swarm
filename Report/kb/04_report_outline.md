# 04 — Report Structure (Agent 4)

Demo analysed: `Report/demo/7th_midterm (3).pdf` (NSUT, B.Tech Project-I, ECE, Sept 2026, 34 pp.).
Adopted: title page layout (B.TECH. PROJECT ON / title / Submitted By + roll numbers / Under the Guidance of /
"Project-I in partial fulfillment of requirement for the award of B.Tech. in Electronics & Communication
Engineering" / Department / NSUT NEW DELHI-110078 / month year); Certificate; Acknowledgement with student
signatures; Plagiarism Report page; Abstract + Keywords; Contents; List of Figures; List of Tables;
chapter heads "CHAPTER n / TITLE" centred; numbered equations (chapter.number); captions "Figure 3.1: ...";
"Results and Discussion" containing "Task, Achievement and Possible Beneficiaries", "Review of Contributions",
"Scope for Future Work"; short Conclusion; numbered bibliography with URLs; Appendix.
Not copied: any IIR-filter technical content.
Added (user request): Candidate's Declaration.

Target: ~24 pages (8 front matter + ~16 body/references).

| Ch | Sections | Figures | Tables |
|---|---|---|---|
| 1 Introduction | 1.1 General, 1.2 Need for the Study, 1.3 Problem Statement, 1.4 Objectives, 1.5 Scope of the Work | – | – |
| 2 Literature Review | 2.1 RL and policy optimisation, 2.2 Multi-agent RL, 2.3 Drone swarms and DRL for quadrotor swarms, 2.4 Imitation-guided RL and set representations, 2.5 Research Gap | – | – |
| 3 Mathematical Modelling and Methodology | 3.1 System overview, 3.2 Problem formulation, 3.3 Flight dynamics and collision model, 3.4 Formation and scenarios, 3.5 Observation and action spaces, 3.6 Reward design, 3.7 Policy network, 3.8 Learning algorithm (imitation warm start, PPO, GAE), 3.9 Evaluation metrics | FIG-01, FIG-02 | env params, observation, reward terms |
| 4 System Design and Implementation | 4.1 Software architecture, 4.2 Training pipeline, 4.3 Training configuration, 4.4 Web-based flight laboratory | FIG-03, FIG-04, FIG-05 | hyperparameters, curriculum |
| 5 Results and Discussion | 5.1 Experimental setup, 5.2 Final demonstration, 5.3 Formation disturbance and recovery, 5.4 Held-out evaluation, 5.5 Training behaviour, 5.6 Ongoing development (V4), 5.7 Task, achievement and possible beneficiaries, 5.8 Review of contributions, 5.9 Scope for future work | FIG-06..FIG-12 | held-out results |
| 6 Conclusion | – | – | – |
| Bibliography, Appendix A (reproduction commands) | | | |
