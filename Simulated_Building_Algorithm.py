import tkinter as tk
from tkinter import ttk
import time
import random
from collections import deque
import datetime
import numpy as np
from sklearn.cluster import KMeans  # for zoning
import logging
from numba import njit
import threading

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

NUM_FLOORS = 20
NUM_ELEVATORS = 4
ELEVATOR_CAPACITY = 15
SIMULATION_SPEED = 1
SKY_LOBBY_FLOOR = 10 #Reduce floors so the GUI doesn't get too long

# Visual Settings
ELEVATOR_WIDTH = 40
ELEVATOR_HEIGHT = 30
SHAFT_WIDTH = 60
CANVAS_WIDTH = NUM_ELEVATORS * SHAFT_WIDTH + 100  # width to account for elevator #
CANVAS_HEIGHT = NUM_FLOORS * 40 #Dynamic

# ENP weights
ENP_WAITING_TIME = 0.4
ENP_DISTANCE = 0.3
ENP_CAPACITY = 0.2
ENP_DIRECTION = 0.1
ENP_VIP = 0.5

# Zoning Configuration
ZONING_UPDATE_INTERVAL = 60  # seconds between zone updates
CLUSTER_LOOKBACK_TIME = 300  # seconds of history to consider for clustering

# -----------------------------------------------------------------------------
# Data Structures
# -----------------------------------------------------------------------------

class Request:
    def __init__(self, request_id, origin_floor, destination_floor, timestamp, passenger_count=1, vip=False):
        self.request_id = request_id
        self.origin_floor = origin_floor
        self.destination_floor = destination_floor
        self.timestamp = timestamp
        self.passenger_count = passenger_count
        self.priority_score = 0
        self.assigned_elevator = None
        self.vip = vip

    def __repr__(self):
        vip_str = " (VIP)" if self.vip else ""
        return f"Req ID: {self.request_id}, Origin: {self.origin_floor}, Dest: {self.destination_floor}, Time: {self.timestamp}{vip_str}"


class Elevator:
    def __init__(self, elevator_id, current_floor=1, direction=0, assigned_zone=None):
        self.elevator_id = elevator_id
        self.current_floor = current_floor
        self.direction = direction  # 0=idle, 1=up, -1=down
        self.passengers = []  # List of Request objects
        self.capacity = ELEVATOR_CAPACITY
        self.destinations = set() # Set of floors this elevator is scheduled to visit
        self.assigned_zone = assigned_zone

    def __repr__(self):
        return f"Elevator {self.elevator_id}: Floor {self.current_floor}, Direction {self.direction}, Passengers: {len(self.passengers)}, Zone: {self.assigned_zone}"

    def is_full(self):
        return len(self.passengers) >= self.capacity

    def add_passenger(self, request):
        self.passengers.append(request)
        self.destinations.add(request.destination_floor)

    def remove_passenger(self, request):
        self.passengers.remove(request)

# -----------------------------------------------------------------------------
# Global Variables
# -----------------------------------------------------------------------------

elevators = [Elevator(i, assigned_zone=i % NUM_ELEVATORS) for i in range(NUM_ELEVATORS)] #initialize with zones
request_queue = deque()
request_id_counter = 0
historical_requests = deque(maxlen=10000) #Store last 10,000 requests
last_zoning_update = time.time()

#GUI data
elevator_rects = {}
floor_labels = {} # store each level text box
elevator_passenger_count = {}

# -----------------------------------------------------------------------------
# ENP Algorithms (Embedded)
# -----------------------------------------------------------------------------
@njit
def enp_fast_string_search(strings, search_term):
    """
    ENP-style string search using explicit looping and Numba for speed.
    This minimizes memory allocations within the search loop.
    """
    n = len(strings)
    matches = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        if search_term in strings[i]:
            matches[i] = True
    return matches

@njit
def enp_distance(x1, y1, x2, y2):
  """Calculates the distance between two points"""
  return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

@njit
def enp_priority(waiting_time, distance, capacity_factor, direction_factor, is_vip, waiting_time_weight, distance_weight, capacity_weight, direction_weight, vip_weight):
  """Calculates priority scores for passenger requests based on ENP criteria."""
  return (waiting_time * waiting_time_weight) + (1 / (distance + 1) * distance_weight) + (capacity_factor * capacity_weight) + (direction_factor * direction_weight) + (is_vip*vip_weight)

# -----------------------------------------------------------------------------
# Data Collection Module
# -----------------------------------------------------------------------------

def generate_random_request():
    """Simulates requests based on time of day, floor usage, and event patterns."""
    global request_id_counter
    request_id_counter += 1
    now = datetime.datetime.now()
    hour = now.hour

    origin_floor = select_origin_floor(hour)
    destination_floor = select_destination_floor(origin_floor, hour)

    timestamp = now
    passenger_count = random.randint(1, 5)
    vip = origin_floor >= 46 or destination_floor >= 46

    new_request = Request(request_id_counter, origin_floor, destination_floor, timestamp, passenger_count, vip)
    historical_requests.append(new_request)
    return new_request

def select_origin_floor(hour):
    """Selects an origin floor based on time of day and building usage."""

    if 7 <= hour < 9:  # Morning Peak
        probabilities = [0.02] * 5 + [0.4] * 15 #Retail, office
    elif 12 <= hour < 13:  # Lunch Peak
        probabilities = [0.3] * 5 + [0.4] * 15 #Retail, office
    elif 17 <= hour < 19:  # Afternoon Peak
        probabilities = [0.05] * 5 + [0.4] * 15 #Retail, office
    elif 19 <= hour < 22:  #Evening
        probabilities = [0.05] * 5 + [0.05] * 15 #Retail, office
    else:  # Off-peak
        probabilities = [0.1] * 5 + [0.1] * 15 #Retail, office

    #Pad to num_floors -1, then normalize
    probabilities = probabilities + [0]*(NUM_FLOORS - len(probabilities))
    probabilities = [p/sum(probabilities) for p in probabilities]

    #Use probabilities as a distribution
    probabilities = [max(p,0) for p in probabilities]
    probabilities = [p/sum(probabilities) for p in probabilities]

    # Use numpy.random.choice for weighted random selection.
    return np.random.choice(range(1, NUM_FLOORS+1), p=probabilities)

def select_destination_floor(origin_floor, hour):
    """Selects a destination floor based on the origin and the time of day."""
    destination_floor = random.randint(1, NUM_FLOORS)
    while destination_floor == origin_floor:
        destination_floor = random.randint(1, NUM_FLOORS)
    return destination_floor

# -----------------------------------------------------------------------------
# Dynamic Zoning Algorithm with KMeans Clustering
# -----------------------------------------------------------------------------

def update_zones():
    """Dynamically adjusts elevator zones based on KMeans clustering of historical requests."""
    global last_zoning_update, request_queue, CLUSTER_LOOKBACK_TIME, ZONING_UPDATE_INTERVAL

    # Perform updates once and a while, or when the reqeust queue is zero
    if time.time() - last_zoning_update < ZONING_UPDATE_INTERVAL and request_queue:
        return  # Skip if not enough time has passed

    last_zoning_update = time.time()

    #Filter requests to only include cluster look back time
    relevant_requests = [req for req in historical_requests if (datetime.datetime.now() - req.timestamp).total_seconds() <= CLUSTER_LOOKBACK_TIME]

    if not relevant_requests:
        logging.info("Not enough data to perform clustering.  Keeping the existing zones")
        return

    # Prepare data for clustering (origin and destination floors)
    floor_data = np.array([[req.origin_floor, req.destination_floor] for req in relevant_requests])

    # Use KMeans to cluster floor data into NUM_ELEVATORS clusters
    kmeans = KMeans(n_clusters=NUM_ELEVATORS, random_state=0, n_init="auto") #Suppress future warning

    try:
        clusters = kmeans.fit_predict(floor_data)
    except Exception as e:
        logging.error(f"KMeans clustering failed: {e}. Keeping existing zones.")
        return

    # Assign each elevator to the centroid of the assigned cluster
    for i, elevator in enumerate(elevators):
        centroid = kmeans.cluster_centers_[i]
        elevator.assigned_zone = round(np.mean(centroid))  # Zone represents average floor in the zone

    logging.info(f"Zones updated: {[e.assigned_zone for e in elevators]}")

# -----------------------------------------------------------------------------
# Prioritization Engine
# -----------------------------------------------------------------------------

def calculate_priority(request, elevators):
    """Calculates priority score, factoring in weights and VIP status."""
    now = datetime.datetime.now()
    waiting_time = (now - request.timestamp).total_seconds()
    closest_elevator = find_closest_elevator(request.origin_floor, elevators)
    distance = abs(closest_elevator.current_floor - request.origin_floor) if closest_elevator else float('inf')
    capacity_factor = 1 if not closest_elevator.is_full() else 0.5
    direction_factor = 1
    is_vip = 2 if request.vip else 1 #VIP multiplier, 2 if vip, 1 if not

    if closest_elevator:
        if closest_elevator.direction == 1 and request.origin_floor > closest_elevator.current_floor:
             direction_factor = 1.2
        elif closest_elevator.direction == -1 and request.origin_floor < closest_elevator.current_floor:
            direction_factor = 1.2
    else:
      print ("There is no elevator")
      return 0

    # Calculate priority score using the imported ENP function
    priority_score = enp_priority(waiting_time, distance, capacity_factor, direction_factor, is_vip, ENP_WAITING_TIME, ENP_DISTANCE, ENP_CAPACITY, ENP_DIRECTION, ENP_VIP)

    request.priority_score = priority_score
    return priority_score

def find_closest_elevator(floor, elevators):
    """Finds the closest available elevator, preferring elevators in the same zone."""
    available_elevators = [e for e in elevators if not e.is_full()]
    if not available_elevators:
        return None

    #Prefer elevators in same zone, then closest elevator
    def elevator_sort_key(elevator):
        zone_match = 1 if elevator.assigned_zone == floor else 0 #prioritize zone matches
        distance = enp_distance(elevator.current_floor, elevator.elevator_id, floor, 1) #calculate enp distance
        return (zone_match, distance) #Tuple automatically sorts by first value, then second
    return min(available_elevators, key=elevator_sort_key)

# -----------------------------------------------------------------------------
# Elevator Control Logic
# -----------------------------------------------------------------------------

def assign_request_to_elevator(request, elevators):
    """Assigns a request, potentially routing through the sky lobby."""
    if (request.origin_floor <= SKY_LOBBY_FLOOR and request.destination_floor > SKY_LOBBY_FLOOR) or \
       (request.origin_floor > SKY_LOBBY_FLOOR and request.destination_floor <= SKY_LOBBY_FLOOR): #Sky lobby transit
        logging.info(f"Request {request.request_id} is using the Sky Lobby Transit.")

        if request.origin_floor <= SKY_LOBBY_FLOOR: #Go to sky lobby
            request.destination_floor = SKY_LOBBY_FLOOR #Temp Dest

            for elevator in elevators:
                calculate_priority(request, elevators)
            best_elevator = find_closest_elevator(request.origin_floor, elevators)

            if best_elevator:
                best_elevator.add_passenger(request)
                request.assigned_elevator = best_elevator.elevator_id
                logging.info(f"Request {request.request_id} assigned to Elevator {best_elevator.elevator_id} (Sky Lobby 1st leg)")
                return True
            else:
                logging.warning(f"No suitable elevator found for request {request.request_id} to Sky Lobby 1st leg.")
                return False
        else: #at sky lobby already
            # Ensure the destination is within valid range
            for elevator in elevators:
                calculate_priority(request, elevators)
            best_elevator = find_closest_elevator(request.origin_floor, elevators)

            if best_elevator:
                best_elevator.add_passenger(request)
                request.assigned_elevator = best_elevator.elevator_id
                logging.info(f"Request {request.request_id} assigned to Elevator {best_elevator.elevator_id} (Sky Lobby 2nd leg)")
                return True
            else:
                logging.warning(f"No suitable elevator found for request {request.request_id} from Sky Lobby 2nd leg.")
                return False

    else: #Not using sky lobby
        for elevator in elevators:
            calculate_priority(request, elevators)

        best_elevator = find_closest_elevator(request.origin_floor, elevators)
        if best_elevator:
            best_elevator.add_passenger(request)
            request.assigned_elevator = best_elevator.elevator_id
            logging.info(f"Request {request.request_id} assigned to Elevator {best_elevator.elevator_id}")
            return True
        else:
            logging.warning(f"No suitable elevator found for request {request.request_id}.")
            return False

def handle_elevator_arrivals():
    """Handles passenger drop-offs, including sky lobby transfers."""
    for elevator in elevators:
        floor = elevator.current_floor
        passengers_to_remove = []
        for request in elevator.passengers:
            if request.destination_floor == floor:
                passengers_to_remove.append(request)

        for request in passengers_to_remove:
            elevator.remove_passenger(request)
            logging.info(f"Request {request.request_id} completed at floor {floor} by Elevator {elevator.elevator_id}")

            if floor == SKY_LOBBY_FLOOR:
              if NUM_FLOORS > SKY_LOBBY_FLOOR+1:
                  request.origin_floor = SKY_LOBBY_FLOOR
                  request.destination_floor = random.randint(SKY_LOBBY_FLOOR + 1, NUM_FLOORS) #Choose a random destination
                  request.assigned_elevator = None
                  request.priority_score = 0
                  logging.info(f"Request {request.request_id} transferred to Sky Lobby.  Destination Floor: {request.destination_floor}")
              else:
                  logging.info(f"Request {request.request_id} can't generate destination higher than floor {NUM_FLOORS}")


        if not elevator.passengers:
            elevator.direction = 0

# -----------------------------------------------------------------------------
# Elevator Movement
# -----------------------------------------------------------------------------

def move_elevators():
    """Simulates the movement of elevators."""
    global elevator_rects, elevator_passenger_count

    for elevator in elevators:
        if elevator.direction == 1:  # Moving Up
            if elevator.current_floor < NUM_FLOORS:
                elevator.current_floor += 1
            else:
                elevator.direction = -1  # Change direction at top floor
        elif elevator.direction == -1:  # Moving Down
            if elevator.current_floor > 1:
                elevator.current_floor -= 1
            else:
                elevator.direction = 1  # Change direction at bottom floor
        else:
            # Elevator is idle. Determine if there is anything it should be doing
            if elevator.destinations:
                # Find the nearest destination
                nearest_destination = min(elevator.destinations, key=lambda x: abs(x - elevator.current_floor))
                if nearest_destination > elevator.current_floor:
                    elevator.direction = 1
                    elevator.current_floor += 1
                elif nearest_destination < elevator.current_floor:
                    elevator.direction = -1
                    elevator.current_floor -= 1

        # Update GUI
        #Move it back ten then add
        update_elevator(elevator)
        time.sleep(1/SIMULATION_SPEED)


# -----------------------------------------------------------------------------
# Simulation
# -----------------------------------------------------------------------------

def run_simulation(duration=3600):  # Run for 1 hour
    """Runs the elevator simulation with dynamic zoning and priority-based dispatching."""
    global elevators

    start_time = time.time()

    while time.time() - start_time < duration:

        # 1. Generate Requests
        if random.random() < 0.3:
            new_request = generate_random_request()
            request_queue.append(new_request)

        # 2. Dynamic Zoning (periodic)
        update_zones()

        # 3. Assign Requests to Elevators
        requests_to_remove = []
        for request in request_queue:
            if request.assigned_elevator is None:
                if assign_request_to_elevator(request, elevators):
                    requests_to_remove.append(request)
        for req in requests_to_remove:
            request_queue.remove(req)

        # 4. Move Elevators
        move_elevators()

        # 5. Handle Elevator Arrivals
        handle_elevator_arrivals()

        time.sleep(1/SIMULATION_SPEED)

    logging.info("Simulation complete.")

# -----------------------------------------------------------------------------
# GUI functions
# -----------------------------------------------------------------------------

def setup_gui():
    """Sets up the Tkinter GUI."""
    global root, canvas, elevator_rects, floor_labels, elevator_passenger_count
    root = tk.Tk()
    root.title("AZDP Elevator Simulation")

    canvas = tk.Canvas(root, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg="white")
    canvas.pack()

    # Draw Elevator Shafts, set rectangle for each of elevator
    for i in range(NUM_ELEVATORS):
        x0 = i * SHAFT_WIDTH + 50
        y0 = 5
        x1 = x0 + ELEVATOR_WIDTH
        y1 = CANVAS_HEIGHT
        canvas.create_rectangle(x0, y0, x1, y1, fill="lightgray", outline="gray")

        #Initialize the elevator at bottom
        elevator_rect = canvas.create_rectangle(x0, (NUM_FLOORS -1) * 40 , x1, (NUM_FLOORS -1) * 40 - ELEVATOR_HEIGHT, fill="blue", outline="black")
        elevator_rects[i] = elevator_rect

    # Draw Floor Lines
    y_start = 5
    for floor in range(1, NUM_FLOORS + 1):
        y = CANVAS_HEIGHT - floor * (CANVAS_HEIGHT/NUM_FLOORS)
        canvas.create_line(0, y, CANVAS_WIDTH, y, fill="black", dash=(4, 4))
        text_id = canvas.create_text(20, y, text=str(floor), anchor=tk.W)

def update_elevator(elevator):
    global elevators, canvas, elevator_rects
    elevator_coord = (NUM_FLOORS - elevator.current_floor) * (CANVAS_HEIGHT/NUM_FLOORS)
    canvas.coords(elevator_rects[elevator.elevator_id], elevator.elevator_id * SHAFT_WIDTH + 50, elevator_coord, elevator.elevator_id * SHAFT_WIDTH + 50 + ELEVATOR_WIDTH, elevator_coord-ELEVATOR_HEIGHT)

def start_simulation():
    """Starts the simulation in a separate thread."""
    run_simulation(duration=100)

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    #Initialize the GUI
    setup_gui()

    #Run the simulation on its own Thread so the UI doesn't get stuck
    simulation_thread = threading.Thread(target=start_simulation, daemon=True)
    simulation_thread.start()

    #Main loop to run it
    tk.mainloop()