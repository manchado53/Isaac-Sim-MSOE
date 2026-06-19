# Nav2 — non-technical presentation script

A speaker-note version of the Nav2 explanation for a general audience.
For the deep-technical reference see
[`NAV2_INTERNALS.md`](NAV2_INTERNALS.md).

Six slides, six visuals.  All PNGs live in
[`../docs/nav2_talk/`](../docs/nav2_talk/) and are regeneratable via
`python3 floorplan_vision/generate_nav2_talk_diagrams.py`.

---

## Slide 1 — What is Nav2?

![Slide 1: overview](../docs/nav2_talk/01_overview.png)

**One sentence:**

> Nav2 is the open-source *brain* that turns a robot with wheels and
> sensors into something that can drive itself from point A to point B
> inside a building without hitting things.

**Analogy they'll all get:**

> Imagine Google Maps combined with a car's autopilot — together, for
> any robot.

**Why they care:** Nav2 is what *we don't have to write*. It's the
shoulders we stand on. Our research goes on top.

---

## Slide 2 — The four roles

![Slide 2: the four roles](../docs/nav2_talk/02_four_roles.png)

Nav2 is not one program — it's four pieces cooperating, same way a
human driver splits their attention:

| Human driver thinks... | Nav2 role | Plain description |
|---|---|---|
| "Where am I right now?" | **Localization (AMCL)** | matches the live lidar scan against the building's map |
| "What should I avoid?"  | **Costmap**              | paints safety halos around every wall and obstacle |
| "What's the route?"     | **Planner**              | computes the shortest walkable path to the goal |
| "How do I steer?"       | **Controller (DWB)**     | actually turns the wheels, moment-by-moment |

**Speaker tip:** walk left-to-right through the boxes, paying attention
to the arrows. Each role hands its output to the next. Failure at any
stage blocks the next one.

---

## Slide 3 — The costmap ("safety halos")

![Slide 3: costmap halo](../docs/nav2_talk/03_costmap_halo.png)

This is the one concept that *clicks* for non-technical audiences.

> The robot mentally paints a colored halo around every wall.
> **Red**    = "will definitely collide"
> **Orange** = "very close, I'd rather not"
> **Yellow** = "getting close, avoid if possible"
> **Green**  = "open space — I'm happy here"

The planner then draws the cheapest route that stays off red and
mostly in green.

**Demo prop:** the left panel of the slide shows a top-down floor
slice with walls (black) and their halos. The right panel is the
legend. Hold on this slide for a moment — it's the key mental model
the rest of the talk relies on.

**Tie-in to the demo:** Every visual in the recorded tour videos —
the red "bad zones" around walls — is this exact costmap rendered in
real time.

---

## Slide 4 — The one tuning slider that matters

![Slide 4: tuning tradeoff](../docs/nav2_talk/04_tuning_tradeoff.png)

All of "configuring Nav2" is balancing one question:
**how fast should the halo fade as you move away from a wall?**

- **Halos fade slowly** → every cell in a narrow corridor looks
  painful to the planner. Robot refuses doorways, crawls, gets stuck.
- **Halos fade fast** → a clear low-cost "spine" runs down the middle
  of narrow corridors. Robot threads doorways happily.

**Story tie-in:** this is *exactly* what blocked us for hours today.
Our initial setting made halos fade very slowly — robot got to one
corridor junction, the entire area looked "painful," and it refused
to plan a route through. Bumped the slider and the junction opened
up.

**Takeaway for the audience:** robotics is full of these trade-offs.
No free lunch — loosen the halos and the robot clips walls in tight
turns; tighten them and the robot refuses doorways.

---

## Slide 5 — What we built on top of Nav2

![Slide 5: the tour story](../docs/nav2_talk/05_tour_story.png)

Our actual research contribution has five layers, none of which are
Nav2 itself:

1. **Floor plan.** Scan the Innovation Center USD (BIM export from
   Revit) to generate a 2D occupancy map.
2. **Nav graph.** Skeletonize walkable space into a graph;
   doors become labeled nodes.
3. **TSP route.** Optimal door visit order via Dijkstra distances +
   2-opt (travelling salesman on a floor plan).
4. **Nav2 tour.** Hand Nav2 the list of poses, let it drive
   door-to-door — we don't reinvent driving.
5. **Record.** Three cameras roll: chase cam, robot POV, top-down
   map overlay — footage feeds the visual anomaly detector.

**Speaker point:** The research is steps 1, 2, 3, and 5 — the
intelligence *surrounding* Nav2. Nav2 is a dependency we stand on,
not a contribution we make.

**Demo moment:** at this point, play ~60 seconds of the hallway-sweep
video
(`tsp_doors_follow_f4_20260419_163937.mp4` — 11 doors in under 2 minutes).

---

## Slide 6 — Summary you can walk away with

![Slide 6: summary](../docs/nav2_talk/06_summary.png)

**One-line takeaway:**

> Nav2  =  Localization + Costmap + Planner + Controller
>
> Standardized driving → we focus on **where to go**,
> **what to record**, **what to detect**.

---

## Words to avoid in this deck

The goal is accessibility. Swap specialist terms for plain language:

| Don't say | Do say |
|---|---|
| "behavior tree" | "step-by-step plan" |
| "costmap inflation" | "safety halos around walls" |
| "DWB critics" | "the driving scores each possible move and picks the best" |
| "A\* pathfinding" | "it finds the shortest walkable route, like Google Maps" |
| "ROS 2" | "the standard robotics communication layer" |
| "lifecycle node" | "a service that can be started/stopped on demand" |

## Rehearsal script (60-second version)

If time is tight and you only get one minute:

> Our robots navigate using **Nav2**, an open-source stack. It does
> four things: knows **where it is**, knows **what to avoid**, plans
> **the route**, and **actually drives**. All four need tuning — for
> example, we spent hours today discovering that a single parameter,
> how fast wall 'safety halos' fade with distance, was making the
> robot refuse doorways. One-line fix. The real research isn't Nav2,
> it's everything we build on top: the building scanner, the nav
> graph, the travelling-salesman route planner, and the camera
> recording pipeline that feeds our visual anomaly detector. Nav2
> gets the robot through the building; we decide **where to go**
> and **what to look for**.

## Assets for the talk

| Need | File |
|---|---|
| Overview slide visual | `docs/nav2_talk/01_overview.png` |
| Four-roles slide visual | `docs/nav2_talk/02_four_roles.png` |
| Costmap-halo slide visual | `docs/nav2_talk/03_costmap_halo.png` |
| Tuning-tradeoff slide visual | `docs/nav2_talk/04_tuning_tradeoff.png` |
| "What we built" slide visual | `docs/nav2_talk/05_tour_story.png` |
| Summary slide visual | `docs/nav2_talk/06_summary.png` |
| TSP/Dijkstra concept (bonus) | `docs/tsp_dijkstra_explainer.png` |
| Demo video (hallway sweep) | `videos/tsp_doors_follow_f4_20260419_163937.mp4` |
| Demo video (onboard POV) | `videos/tsp_doors_onboard_f4_20260419_163937.mp4` |
| Demo video (top-down map) | `videos/tsp_doors_map_f4_20260419_163937.mp4` |
| Technical deep-dive (for Q&A backup) | `navigation/NAV2_INTERNALS.md` |

## Likely Q&A — prepped answers

**"How did you make the robot work?"** — Nav2 is open-source. We didn't
build it; we configured it. Tuned maybe a dozen parameters over the
course of the project.

**"Why not just use a gamepad / waypoint clicker?"** — We want the
robot to handle unexpected obstacles (a person walking into a corridor,
a box someone left out). Nav2's controller replans in real time; a
recorded path would just crash into whatever's new.

**"Is it perfect?"** — No. Some doors on floor 4 aren't reachable
because they sit 1 cm from a wall and Nav2's safety margins won't let
the robot stop there. Real robots have real limits. Our automation
pipeline filters these out.

**"How much of this is simulation vs. real?"** — Today 100 %
simulation (NVIDIA Isaac Sim, NVIDIA GPU). But Nav2 is the same stack
used on physical robots — the tuning transfers. That's the whole
point of BIM-to-sim: we can develop + validate the full pipeline
*before* the physical building is finished.

**"Can this scale?"** — Yes. The pipeline is building-agnostic: feed
a new Revit export, run the four-stage pipeline (scan → graph → doors
→ TSP), and Nav2 handles the driving. The occupancy scanner produced
usable maps for all four floors of the Innovation Center with no
per-floor tuning.
