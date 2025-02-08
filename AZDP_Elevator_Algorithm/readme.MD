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

### 3. Algorithms  

#### 3.1 Traffic Simulation  
Passenger requests are generated using a **probabilistic model** with factors:  
- **Time-of-Day Profiles**: Morning, lunch, afternoon, evening, and off-peak hours.  
- **Floor Type**: Implicit classification based on a fixed probability distribution.  
**Example**: During morning peak, higher probability for requests from lower floors (e.g., apartments → offices).  

#### 3.2 Dynamic Zoning  
Dynamic elevator zones are determined using **K-Means clustering** on historical request data:  
1. **Data Collection**: Origin-destination pairs from recent requests.  
2. **Feature Extraction**: Requests as 2D points (`x = origin_floor`, `y = destination_floor`).  
3. **Clustering**: K-Means with `NUM_ELEVATORS` clusters.  
4. **Zone Assignment**: Elevators assigned to cluster centroids.  
**Update Interval**: Configurable (`ZONING_UPDATE_INTERVAL`).  

**Euclidean Distance Formula**:  
\[
d = \sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2}
\]  
where \((x_1, y_1)\) and \((x_2, y_2)\) are elevator and floor coordinates, respectively.  

#### 3.3 Prioritization  
Each request's priority score (\(P\)) is a **weighted sum** of factors:  
\[
P = w_t \cdot t + \frac{w_d}{d + 1} + w_c \cdot C + w_{dir} \cdot D + w_{vip} \cdot V
\]  
where:  
- \(w_t\), \(w_d\), \(w_c\), \(w_{dir}\), \(w_{vip}\): Weights for waiting time, distance, capacity, direction, and VIP status.  
- \(t\): Waiting time.  
- \(d\): Distance to the closest elevator.  
- \(C\): Capacity factor (1 if not full, 0.5 if full).  
- \(D\): Direction factor (1.2 if matching, else 1).  
- \(V\): VIP status (1 if VIP, else 0).  

**ENP Distance Calculation**:  
\[
d = \sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2}
\]  
(Implemented with `numba` for performance.)  

#### 3.4 Dispatching  
Request assignment to elevators follows:  
1. Calculate priority scores for each available elevator.  
2. Prefer elevators in the same zone.  
3. Select the elevator with the highest priority score.  

---

### 4. ENP Techniques  
The simulation leverages **Numba JIT compilation** for performance-critical code (e.g., priority calculation) and **data locality optimizations**.  

---

### 5. Usage  
1. Install dependencies:  
   ```bash
   git clone [repository URL]
   cd AZDP_Elevator_Algorithm
   pip install numpy scikit-learn matplotlib numba tkinter
   python Simulated_Building_Algorithm.py