# AZDP Elevator Control System: A Simulation Framework for Adaptive Zoning and Dynamic Prioritization

[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/numpy-%5E1.20-green)](https://numpy.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-%5E0.24-orange)](https://scikit-learn.org/)
[![matplotlib](https://img.shields.io/badge/matplotlib-%5E3.4-purple)](https://matplotlib.org/)
[![Numba](https://img.shields.io/badge/numba-%5E0.53-yellow)](https://numba.pydata.org/)
[![Tkinter](https://img.shields.io/badge/tkinter--blue)](https://docs.python.org/3/library/tkinter.html)

---

### Abstract  
This document outlines a **Tkinter-based simulation framework** for an **Adaptive Zoning and Dynamic Prioritization (AZDP)** elevator control system designed for high-rise, mixed-use buildings. The system integrates **ENP** system(view ) to optimize traffic flow, dynamically adjust elevator zones, and prioritize passenger requests using a weighted scoring system.  

### 1. Introduction  
Elevator systems in modern high-rise buildings face challenges such as minimizing wait times, optimizing resource allocation, and handling varying traffic patterns. The AZDP system addresses these by:  
- **Dynamic zoning**: Clustering requests to assign elevators to specific zones.  
- **Dynamic prioritization**: Calculating priority scores for requests based on waiting time, distance, elevator capacity, direction, and VIP status.  

---

### 2. System Architecture  
The AZDP simulation models the following components:  
- **Building Model**: Configurable number of floors (`NUM_FLOORS`) and mixed-use profiles.  
- **Elevator Model**: Set of elevators (`NUM_ELEVATORS`) with capacity (`ELEVATOR_CAPACITY`).  
- **Traffic Generator**: Simulates passenger requests based on time-of-day profiles and floor usage.  
- **Dynamic Zoning Algorithm**: K-Means clustering using `scikit-learn` to assign elevator zones.  
- **Prioritization Engine**: Calculates priority scores using weighted factors.  
- **Dispatching Logic**: Assigns elevators using proximity, score, and zone association.  
- **Simulation Engine**: Processes requests and moves elevators in discrete time steps.  
- **Tkinter GUI**: Basic visualization of simulation progress.  

---

To ensure proper rendering of mathematical equations in your GitHub Markdown README, replace \[...\] with $$...$$ and remove indentation from the equations. Here's the corrected version:
markdownCopy
# AZDP Elevator Control System: A Simulation Framework for Adaptive Zoning and Dynamic Prioritization

[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/numpy-%5E1.20-green)](https://numpy.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-%5E0.24-orange)](https://scikit-learn.org/)
[![matplotlib](https://img.shields.io/badge/matplotlib-%5E3.4-purple)](https://matplotlib.org/)
[![Numba](https://img.shields.io/badge/numba-%5E0.53-yellow)](https://numba.pydata.org/)
[![Tkinter](https://img.shields.io/badge/tkinter--blue)](https://docs.python.org/3/library/tkinter.html)

---

## Abstract
This document outlines a **Tkinter-based simulation framework** for an **Adaptive Zoning and Dynamic Prioritization (AZDP)** elevator control system for high-rise, mixed-use buildings. The system uses **Enhanced Numerical Performance (ENP)** techniques for traffic optimization, dynamic zoning, and prioritization.

---

## 3. Algorithms

### 3.1 Traffic Simulation
Passenger requests are generated probabilistically using time-of-day (rush hour, lunch, etc.) and floor type (apartment vs. office).

### 3.2 Dynamic Zoning
Dynamic elevator zones are determined via **K-Means clustering**:
1. **Data Collection**: Historical request data (origin/destination floors) is collected.
2. **Feature Extraction**: Requests are represented as points in 2D space (origin and destination).
3. **Clustering**: Cluster data into `NUM_ELEVATORS` zones using K-Means.
4. **Zone Assignment**: Elevators are assigned to centroids of clusters.
5. **Periodic Update**: Zones update every `ZONING_UPDATE_INTERVAL` seconds.

### 3.3 Prioritization
Each request's priority score ($P$) is a weighted sum of factors:
$$
P = w_t \cdot t + \frac{w_d}{d + 1} + w_c \cdot C + w_{dir} \cdot D + w_{vip} \cdot V
$$
where:
- $w_t, w_d, w_c, w_{dir}, w_{vip}$: Weights for waiting time, distance, capacity, direction, and VIP status.
- $t$: Waiting time (seconds).
- $d$: Distance to closest elevator (floors).
- $C$: Capacity factor (1 if not full, 0.5 if full).
- $D$: Direction factor (1.2 if elevator matches passenger direction, else 1).
- $V$: VIP status (1 if true).

The distance formula is:
$$
d = \sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2}
$$

### 3.4 Dispatching
Requests are assigned elevators using priority scores, preferring same-zone elevators.

---

## 4. Usage
- Clone the repository:
  ```bash
  git clone https://github.com/Ziqian-Huang0607/AZDP_Elevator_Algorithm
  cd AZDP_Elevator_Algorithm
Install dependencies:

` pip install numpy scikit-learn matplotlib numba tkinter `
Configure parameters in Simulated_Building_Algorithm.py (e.g., NUM_FLOORS, ENP_weights).

Run the simulation:
` python Simulated_Building_Algorithm.py `

6. Real-World Considerations
Robustness: Handle outages and failures.
Hardware Integration: Implement on microcontrollers with low-level code.
Certification: Comply with safety regulations.
Security: Secure control interfaces.
Copy
