import os
import pygame
from time import sleep
from ultralytics import YOLO
from PIL import Image

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 800
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Traffic Signal Simulation")

# Colors
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)

# Font for text
font = pygame.font.Font(None, 36)

# Load YOLO model
model = YOLO("/Users/my/Desktop/FYP2/custom_yolo_finetuned2/weights/best.pt")

# Constants
minGreenTime = 10   # Minimum green light duration (seconds)
maxGreenTime = 120  # Maximum green light duration (seconds)
averageTimeOfClass = {"bus": 10, "car": 10, "jeepney": 6, "motorcycle": 3, "tricycle": 4, "truck": 12, "van": 7}
trafficDensityWeight = {"Low": 1, "Medium": 1.5, "High": 2}  # Density weight categories
helmetless_dir = "/Users/my/Desktop/FYP2/helmetV"
os.makedirs(helmetless_dir, exist_ok=True)  # Directory to save helmet violations

def calculateTrafficDensityWeight(vehicleCount, laneCapacity):
    density = vehicleCount / laneCapacity
    if density < 0.83:
        return "Low"
    elif density < 1.67:
        return "Medium"
    else:
        return "High"

def calculateGreenSignalTime(vehicleData, noOfLanes, laneCapacity):
    totalWeightedTime = sum(vehicleData.get(v, 0) * averageTimeOfClass.get(v, 5) for v in vehicleData)
    densityCategory = calculateTrafficDensityWeight(sum(vehicleData.values()), laneCapacity)
    densityWeight = trafficDensityWeight[densityCategory]
    return max(minGreenTime, min((totalWeightedTime * densityWeight) / (noOfLanes + 1), maxGreenTime))

def draw_signal_with_timer(direction, color, remaining_time):
    positions = {"West": (100, SCREEN_HEIGHT // 2), "East": (700, SCREEN_HEIGHT // 2), "North": (SCREEN_WIDTH // 2, 100), "South": (SCREEN_WIDTH // 2, 700)}
    pos = positions[direction]
    pygame.draw.circle(screen, color, pos, 50)
    screen.blit(font.render(f"{int(remaining_time)}s", True, (0, 0, 0)), (pos[0] - 20, pos[1] - 70))

def display_image(image_path, position):
    img = pygame.image.load(image_path)
    img = pygame.transform.scale(img, (300, 200))
    screen.blit(img, position)

def process_detections(image_path, results, direction):
    vehicle_counts = {"bus": 0, "car": 0, "jeepney": 0, "motorcycle": 0, "tricycle": 0, "truck": 0, "van": 0}
    helmet_violations = False
    with_helmet_count, without_helmet_count = 0, 0
    image = Image.open(image_path)
    violation_counter = 0

    for box in results.boxes:
        cls_name = results.names[int(box.cls)]
        confidence = box.conf.item()  # Get detection confidence
        xyxy = box.xyxy.tolist()[0]  # Get bounding box coordinates

        # Scale bounding box coordinates to match the displayed image size
        scale_x = 300 / image.width  # Scaling factor for width
        scale_y = 200 / image.height  # Scaling factor for height
        scaled_xyxy = [xyxy[0] * scale_x + 250, xyxy[1] * scale_y + 300, xyxy[2] * scale_x + 250, xyxy[3] * scale_y + 300]

        # Draw bounding box and confidence on the image
        pygame.draw.rect(screen, (0, 255, 0), (scaled_xyxy[0], scaled_xyxy[1], scaled_xyxy[2] - scaled_xyxy[0], scaled_xyxy[3] - scaled_xyxy[1]), 2)
        screen.blit(font.render(f"{cls_name} {confidence:.2f}", True, (0, 255, 0)), (scaled_xyxy[0], scaled_xyxy[1] - 20))

        if cls_name in vehicle_counts:
            vehicle_counts[cls_name] += 1
        if cls_name == "without helmet":
            helmet_violations = True
            without_helmet_count += 1
            violation_counter += 1
            cropped_region = image.crop(map(int, xyxy))
            save_path = os.path.join(helmetless_dir, f"{direction}_helmet_violation_{violation_counter}_{os.path.basename(image_path)}")
            cropped_region.save(save_path)
        if cls_name == "with helmet":
            with_helmet_count += 1
    vehicle_counts["Motorcycle"] = with_helmet_count + without_helmet_count
    return vehicle_counts

def display_calculations(vehicle_data, green_signal_time, direction, density_category):
    # Display traffic signal calculations
    for idx, text in enumerate([f"Direction: {direction}", f"Traffic Density: {density_category}", f"Green Signal Time: {green_signal_time:.2f} s"]):
        screen.blit(font.render(text, True, (0, 0, 0)), (20, 120 + idx * 40))

    # Display vehicle counts at the bottom of the screen
    y_offset = SCREEN_HEIGHT - 200
    for vehicle, count in vehicle_data.items():
        screen.blit(font.render(f"{vehicle}: {count}", True, (0, 0, 0)), (20, y_offset))
        y_offset += 40

def main():
    iteration_images = [
        {"West": "/Users/my/Desktop/FYP2/timages/p10.png", "East": "/Users/my/Desktop/FYP2/timages/p4.jpg", "North": "/Users/my/Desktop/FYP2/timages/p8.jpg", "South": "/Users/my/Desktop/FYP2/timages/p2.jpg"},
        {"West": "/Users/my/Desktop/FYP2/timages/p6.jpg", "East": "/Users/my/Desktop/FYP2/timages/p3.jpg", "North": "/Users/my/Desktop/FYP2/timages/p5.jpg", "South": "/Users/my/Desktop/FYP2/timages/p7.jpg"}
    ]
    for iteration, image_set in enumerate(iteration_images, start=1):
        print(f"\n--- Iteration {iteration} ---")
        for direction, image_path in image_set.items():
            results = model(image_path)[0]
            vehicle_counts = process_detections(image_path, results, direction)
            green_signal_time = calculateGreenSignalTime(vehicle_counts, 4, 6)  # Updated noOfLanes=4, laneCapacity=6
            density_category = calculateTrafficDensityWeight(sum(vehicle_counts.values()), 6)
            running, start_time, signal_state = True, pygame.time.get_ticks(), "green"
            green_time_remaining, yellow_time_remaining, red_time_remaining = green_signal_time, 2, 5
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                screen.fill(WHITE)
                if signal_state == "green":
                    draw_signal_with_timer(direction, GREEN, green_time_remaining)
                    green_time_remaining = green_signal_time - (pygame.time.get_ticks() - start_time) / 1000
                    if green_time_remaining <= 0:
                        signal_state, start_time = "yellow", pygame.time.get_ticks()
                elif signal_state == "yellow":
                    draw_signal_with_timer(direction, YELLOW, yellow_time_remaining)
                    yellow_time_remaining -= (pygame.time.get_ticks() - start_time) / 1000
                    if yellow_time_remaining <= 0:
                        signal_state, start_time = "red", pygame.time.get_ticks()
                elif signal_state == "red":
                    red_time_remaining -= (pygame.time.get_ticks() - start_time) / 1000
                    running = False
                display_image(image_path, (250, 300))
                display_calculations(vehicle_counts, green_signal_time, direction, density_category)
                pygame.display.flip()
                pygame.time.wait(1000)
    pygame.quit()

main()