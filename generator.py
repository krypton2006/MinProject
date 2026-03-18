import time
import random
from datetime import datetime
print("STARTED GENERATOR")
def generate_heartbeat():
    # Base normal range
    hb = random.randint(65, 90)

    # Add smooth variation
    hb += random.randint(-5, 5)

    # Rare anomalies
    chance = random.random()

    if chance < 0.05:
        return random.randint(110, 140)  # high spike
    elif chance < 0.10:
        return random.randint(40, 55)   # low drop

    return hb

print("Generating heartbeat data... Press CTRL+C to stop")

while True:
    hb = generate_heartbeat()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Save to file
    with open("heartbeat.csv", "a") as file:
        file.write(f"{timestamp},{hb}\n")

    print(f"{timestamp} → {hb}")

    time.sleep(1)   # 1 second interval
