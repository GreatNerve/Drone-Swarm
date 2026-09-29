# 02 — Theory, Mathematics and Literature (Agent 2)

Input: 00_orchestrator_brief.md, 01_project_knowledge_base.md. Bibliography: `Report/latex/references/references.bib`.
Only concepts actually used by the code are included. Each item: relevance -> equation -> code -> source -> report location.

## Concepts selected for the report

| # | Concept | Why relevant (code) | Equation in report | Source key | Section |
|---|---|---|---|---|---|
| M1 | Decentralised partially observable multi-agent MDP | each drone acts on local obs; shared reward structure | tuple (N, S, {O_i}, A, P, r_i, gamma); policy pi_theta(a_i|o_i) | sutton2018reinforcement, yu2022surprising | 3.1 |
| M2 | Discounted return & value | GAE/critic | G_t = sum gamma^k r_{t+k}; V(o) = E[G_t|o] | sutton2018reinforcement | 3.1 |
| M3 | Stabilised velocity dynamics | env.step | u = limit_3(3a); a_cmd = limit_5((u-v)/tau); v+=(a_cmd+w)h; p+=v h | project (panerati2021learning motivates abstraction) | 3.2 |
| M4 | Swept collision check | env._clearance | min over segment of ||d0 + s(d1-d0)||, s in [0,1] minus radii | vandenberg2011reciprocal (relative motion) | 3.2 |
| M5 | Slot / formation definition | env.slots | s_i(t) = q(t) - (d_s,0,0) + delta_i | batra2021decentralized | 3.3 |
| M6 | Observation space | env._observe | o_i = (x_self in R^17, X_nb in R^{6x8}, X_ob in R^{6x8}) | batra2021decentralized, huang2024collision | 3.4 |
| M7 | Reward function | env._reward | full cost C and r = dt(1-C) - 100·1[culprit] | batra2021decentralized (structure), project coefficients | 3.5 |
| M8 | Permutation-invariant set encoding with attention pooling | policy.SetEncoder | z = MHA(q, phi(X), phi(X); mask) | zaheer2017deep, lee2019set, vaswani2017attention | 3.6 |
| M9 | Tanh-squashed Gaussian policy & log-prob correction | policy.log_probability | log pi(a)=sum[log N(x;mu,sigma) - log(1-tanh^2 x)], with stable form 2(log2 - x - softplus(-2x)) | haarnoja2018soft | 3.6 |
| M10 | Parameter sharing, IPPO with local critic | train.algorithm string | same theta for all drones; V_phi(o_i) | dewitt2020independent, yu2022surprising | 3.6 |
| M11 | Imitation warm start: DART + DAgger | train.clone | L_BC = E ||tanh(mu_theta(o)) - a*(o)||^2 ; D <- D ∪ {(o, a*(o)) : o ~ d_{pi_theta}} | ross2011reduction, laskey2017dart | 3.7 |
| M12 | PPO clipped surrogate | train.py L507-512 | L_clip = -E min(rho A, clip(rho,1-eps,1+eps) A) | schulman2017proximal | 3.7 |
| M13 | GAE with time-limit bootstrap | train.advantages | delta_t = r_t + gamma V(o_{t+1})(1-d_t) - V(o_t); A_t = delta_t + gamma lambda (1-e_t) A_{t+1} | schulman2016gae, pardo2018time | 3.7 |
| M14 | Combined loss with decaying demonstration term | train.py L515-522 | L = L_clip + 0.5 L_V + beta(f) L_BC, beta: 1.0->0.5 | rajeswaran2018learning, schmitt2018kickstarting | 3.7 |
| M15 | Implementation details (reward scaling, adv. norm., grad clip, KL stop, lr anneal, Adam eps) | train.py | text + table only | andrychowicz2021matters, huang2022ppodetails | 3.7 / Table |
| M16 | Curriculum learning | curriculum.STAGES | text | bengio2009curriculum | 3.8 |
| M17 | Evaluation metrics: slot RMSE, recovery, selection score, seed separation | env.summary, evaluate | RMSE formula; recovery condition; score formula | cobbe2020leveraging (seed splits) | 3.9 |

Excluded (not used): Q-learning / Bellman optimality for Q, MADDPG algorithm (only CTDE concept, optional central critic is off), ORCA linear program (not implemented), MPPI (disabled), full Set Transformer (only pooling), value normalisation of MAPPO (not implemented).

## Exact equations to use (project notation)

- Dynamics (per 10 ms substep, 5 per 50 ms control step), tau = 0.28 s, a_max = 5 m/s^2, v_max = 3 m/s.
- limit_b(x) = x·min(1, b/||x||).
- Slot error e_i = ||p_i - s_i||.
- Reward: C_slot = e_c^2 (e_c<=1), 2e_c - 1 (e_c>1), e_c = min(e,5);
  C = C_slot + 0.12 min(b,3)^2 + 6 [0.45 - g_nn]_+^2 + 3 min(sum_o [1 - g_o/1]_+^2, 2) + 0.005||u||^2 + 0.001 min(||Δu||/Δt,20)^2;
  r_{i,t} = Δt(1 - C_{i,t}) - 100·1[i involved in failure].
- Attention pooling: h_j = phi(x_j / scale) (MLP 7->64->128, tanh); z = softmax((q W_Q)(H W_K)^T / sqrt(d_k) + M) (H W_V) W_O with 4 heads, M = -inf on padded rows; empty set -> 0.
- Policy: x ~ N(mu_theta(o_i), diag sigma^2), a_i = tanh(x); eval action tanh(mu).
- PPO ratio rho = exp(log pi_theta(x|o) - log pi_old(x|o)).
- Value loss L_V = 0.5·E[(V_phi(o) - R_hat)^2], total coefficient 0.25 on MSE (0.5 × 0.5) [reference_context history.md #7].
- beta(f) = 1.0 + (0.5 - 1.0) f, lr(f) = lr0 (1 - 0.9 f), f = stage progress.
- Selection score S = mean_s(success_s) - 0.25 mean_s(min(RMSE_s, 2)) - 0.25(1 - recovered/events).
- Slot RMSE = sqrt( (1/T_e) sum_t Δt (1/N) sum_i ||p_i - s_i||^2 ).
- Recovery: max_i ||p_i - s_i|| <= 0.35 m and max_i ||v_i - v_q|| <= 0.35 m/s held for 1 s after object exit.

## Literature map (for Chapter 2) — all in references.bib and all to be cited

- Swarm background: reynolds1987flocks (flocking rules), chung2018survey (aerial swarm survey).
- DRL for drones: kaufmann2023champion (DRL drone racing), panerati2021learning (gym-pybullet-drones).
- DRL for quadrotor swarms: batra2021decentralized (shared decentralised policy, attention over neighbours, formation re-establishment around moving obstacles), huang2024collision (obstacle observations, attention, replay from pre-collision states), huang2023quadswarm (modular simulator, randomised scenarios).
- MARL algorithms: lowe2017multi (CTDE), dewitt2020independent (IPPO), yu2022surprising (MAPPO / PPO in cooperative games), terry2021pettingzoo (API).
- Policy optimisation: schulman2017proximal, schulman2016gae, pardo2018time, andrychowicz2021matters, huang2022ppodetails, haarnoja2018soft.
- Set representations: zaheer2017deep, lee2019set, vaswani2017attention.
- Imitation + RL: ross2011reduction, laskey2017dart, rajeswaran2018learning, schmitt2018kickstarting.
- Curriculum/evaluation: bengio2009curriculum, cobbe2020leveraging.
- Relative-motion collision avoidance: vandenberg2011reciprocal.
- Tooling: paszke2019pytorch, sutton2018reinforcement.

Research gap (grounded in research-plan.md, phrased positively): existing swarm-RL work (Batra 2021, Huang 2024)
reports large training budgets (~1e9 samples in Batra) with direct thrust control; this project studies a
laptop-scale pipeline where a shared policy learns coordination above a stabilised flight layer, with an
explicit same-slot recovery requirement and held-out evaluation.
