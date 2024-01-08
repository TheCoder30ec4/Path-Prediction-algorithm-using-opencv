import cv2
import numpy as np
from queue import PriorityQueue

# Define the check_obstacle function
def check_obstacle(node1, node2, grid):
    x1, y1 = node1
    x2, y2 = node2
    # Check if any of the pixels in the line between the two nodes is black (i.e. an obstacle)
    for x, y in zip(np.linspace(x1, x2, num=100), np.linspace(y1, y2, num=100)):
        if grid[int(y), int(x)] == 0:
            return True
    return False

# Define the heuristic function
def heuristic(node1, node2):
    x1, y1 = node1
    x2, y2 = node2
    # Calculate the Euclidean distance between the two nodes
    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return distance

# Define the A* algorithm
def astar(start, end, grid):
    queue = PriorityQueue()
    print("hello")
    queue.put((0, start))
    g_scores = {start: 0}
    paths = {start: [start]}
    while not queue.empty():
        # Get the node with the lowest f score
        current = queue.get()[1]
        # Check if the current node is the end node
        if current == end:
            return paths[current]
        # Loop through the neighbors of the current node
        for neighbor in [(current[0]+1, current[1]), (current[0]-1, current[1]), (current[0], current[1]+1), (current[0], current[1]-1)]:
            # Check if the neighbor is within the grid
            if neighbor[0] < 0 or neighbor[0] >= grid.shape[1] or neighbor[1] < 0 or neighbor[1] >= grid.shape[0]:
                continue
            # Calculate the tentative g score for the neighbor
            tentative_g_score = g_scores[current] + heuristic(current, neighbor)
            # Check if the neighbor is not an obstacle
            if not check_obstacle(current, neighbor, grid):
                # Check if the neighbor has not been visited before or if the tentative g score is lower than the previous one
                if neighbor not in g_scores or tentative_g_score < g_scores[neighbor]:
                    # Update the g score and add the neighbor to the queue
                    g_scores[neighbor] = tentative_g_score
                    f_score = tentative_g_score + heuristic(neighbor, end)
                    queue.put((f_score, neighbor))
                    # Update the path to the neighbor
                    paths[neighbor] = paths[current] + [neighbor]
    # If there is no path from start to end, return an empty list
    return []

# Load the input image
img = cv2.imread("C:/Users/Seshu Reddy/Desktop/1_RGB_Frames/1.png")

# Convert the input image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Convert the grayscale image to a binary image
binary = cv2.threshold(gray, 55, 255, cv2.THRESH_BINARY)[1]

# Define the start and end points
start = (620, 450)
end = (280, 268)

# Run the A* algorithm
path = astar(start, end, binary)

# Draw the path on the input image
for node in path:
    cv2.circle(img, node, 1, (0, 255, 0), -1)

# Show the output image
cv2.imwrite('output_image_with_paths.jpg', img)
cv2.imshow('Paths', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
