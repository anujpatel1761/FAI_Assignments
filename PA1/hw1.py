import time
import numpy as np
from gridgame import *  # Import all functions and variables from gridgame
import random
import math

"""
Code workflow:
For example, I have a 10x10 grid** with **five shapes already colored. Program begins by recognizing these pre-colored cells and setting them aside. 
Using the Simulated Annealing technique, it then assigns colors to the remaining empty cells, making sure that no two adjacent cells share the same color. 
The code use efftctive method  colors to minimize conflicts, effectively finding an optimal coloring solution. Once the grid is properly colored, 
the program moves on to place additional shapes to cover the colored areas efficiently. 
It does this by selecting the largest possible shape that fits each target spot, 
which helps in reducing the total number of shapes used. The brush navigates to each chosen position
, sets the correct shape and color, and places it on the grid. After all placements, the program verifies that the entire grid meets the coloring rules and calculates 
important statistics like the number of shapes used, the time taken, and any remaining unfilled cells. 
This streamlined process ensures that grid is fully and efficiently colored with minimal colors and shapes, adhering to all the given constraints.
"""
##############################################################################################################################

setup(GUI=False, render_delay_sec=0.1, gs=10)

##############################################################################################################################

shapePos, currentShapeIndex, currentColorIndex, grid, placedShapes, done = execute('export')
print(shapePos, currentShapeIndex, currentColorIndex, grid, placedShapes, done)

####################################################
# Timing your code's execution for the leaderboard.
####################################################

start = time.time()  # <- do not modify this.

##########################################
# Write all your code in the area below.
##########################################

#Convert grid to a NumPy array of integers
# grid = np.array(grid, dtype=int)
# print("my grid is/n",grid)


# Calculate the number of pre-colored cells before placing shapes
pre_colored_cells = np.count_nonzero(grid != -1)
#print("Number of pre olored cells are:\n",pre_colored_cells)

# Define the grid size and maximum number of colors
gridSize = len(grid)
#print("greed size is\n",gridSize)
max_colors_available = len(colors)
#print("color availbale\n",max_colors_available)


"""
Using a separate `color_assignment` matrix instead of the direct grid allows for flexible manipulation and testing without altering the initial grid setup. 
This is useful for algorithms that require frequent adjustments, like simulated annealing. 
It provides a clear framework for operations like color changes, algorithm testing, and state comparisons, 
"""
# Initialize the color assignment matrix with -1 (unassigned)
color_assignment = np.full((gridSize, gridSize), -1, dtype=int)
#print("color_assignment", color_assignment)

# Pre-fill the color assignment matrix with initial colors from the grid
for y in range(gridSize):
    for x in range(gridSize):
        if grid[y, x] != -1:  # Check if the cell in the original grid has a pre-assigned color
            color_assignment[y, x] = grid[y, x]  # Copy that color into the color_assignment matrix
#print("Final color_assignment after filling pre-colored cells:\n", color_assignment)

###################################################################################################################################################

def move_brush_to(target_pos):
    """
    Moves the brush to a specified position on the grid.
    Inputs:
    target_pos: A list or tuple [x, y] representing the target coordinates.
    Outputs:
    None. The function updates the global variable 'shapePos' to the new position.
    Purpose:
    This function is designed to navigate the brush to a specific location on the grid. 
    for placing shapes accurately during the grid coloring process. 
    How it works:
    First, it calculates the horizontal (x) and vertical (y) differences between the current brush position and the target position.
    These differences guide the direction and quantity of movements required to reach the target.
    Example Usage:
    To place a shape at a specific grid location, I would first move the brush to the desired start position using this function.
    """
    global shapePos
    #print("shapePos",shapePos)
    x_diff = target_pos[0] - shapePos[0]
    y_diff = target_pos[1] - shapePos[1]
    # Move horizontally
    for _ in range(abs(x_diff)):
        if x_diff > 0:
            execute('right')
        else:
            execute('left')
        shapePos, _, _, _, _, _ = execute('export')
    # Move vertically
    for _ in range(abs(y_diff)):
        if y_diff > 0:
            execute('down')
        else:
            execute('up')
        shapePos, _, _, _, _, _ = execute('export')
###################################################################################################################################################

def set_shape(shape_index):
    """
    Sets the current brush shape to the desired shape index.
    Inputs:
    shape_index: An integer representing the desired shape index (0-8).
    Outputs:
    The function updates the global variable 'currentShapeIndex' to the new shape index.
    Purpose:
    To select the appropriate brush shape before placing it on the grid.
    How it works:
    The function continually issues the 'switchshape' command until the 'currentShapeIndex' matches the specified 'shape_index'.
    This loop ensures that regardless of the current shape, the brush can be set to any required shape, enhancing the tool's flexibility.
    The 'switchshape' command cycles through available shapes. After each command, it retrieves and updates the global state,
    checking if the desired shape is reached, thereby aligning the brush with the required drawing or interaction specification.
    """
    global currentShapeIndex
    while currentShapeIndex != shape_index:
        execute('switchshape')
        shapePos, currentShapeIndex, _, _, _, _ = execute('export')
        
###################################################################################################################################################

def set_color(color_index):
    """
    Sets the current brush color to the desired color index.
    Inputs:
    color_index: An integer representing the desired color index (0-3).
    Outputs:
    This function updates the 'currentColorIndex' to reflect the new color selection.
    Purpose:
    To select the appropriate color before placing a shape.
    How it works:
    The function enters a loop that continues until the current color of the brush matches the desired 'color_index'.
    Within the loop, it issues a 'switchcolor' command which cycles through the available colors. After each switch, it checks if the selected color matches the desired one.
    It updates the global 'currentColorIndex' each time the color is switched, ensuring that the brush is set to the correct color before any drawing action is performed.
    """
    global currentColorIndex # track the current color of the brush.
    # Keep switching the brush color until it matches the desired one
    while currentColorIndex != color_index:
        execute('switchcolor') # Change to the next color
        shapePos, _, currentColorIndex, _, _, _ = execute('export')
            
###################################################################################################################################################

def get_adjacent_cells(y, x):
    """
    Returns a list of adjacent cell coordinates for a given cell.
    Inputs:
    y-row, x-column
    Outputs:
    adjacent: A list of tuples representing the coordinates of adjacent cells.
    Purpose:
    figure out which cells touch a particular cell on the grid. 
    How it works:
    The function checks each of the four possible neighboring positions (left, right, above, below) to see if they exist within the boundaries of the grid.
    It adds the coordinates of each valid neighbor to the list of adjacent cells.
    """
    adjacent = []
    if x > 0:
        adjacent.append((y, x - 1))
    if x < gridSize - 1:
        adjacent.append((y, x + 1))
    if y > 0:
        adjacent.append((y - 1, x))
    if y < gridSize - 1:
        adjacent.append((y + 1, x))
    return adjacent

###################################################################################################################################################

def calculate_conflicts(color_assignment):
    """
    Calculates the number of conflicts in the current color assignment.
    Inputs:
    color_assignment: A 2D NumPy array representing the color of each cell.
    Outputs:
    conflicts: The total count of adjacent pairs of cells with the same color.
    Purpose:
    To evaluate how many adjacent cells have the same color, which violates the coloring constraint.
    How it works:
    It loops through each cell in the grid.
    For each cell that has a defined color (not -1, which means unassigned), it checks all neighboring cells.
    If a neighbor has the same color, this is counted as a conflict.
    Since each pair of conflicting cells gets counted twice (once for each cell in the pair), the final count of conflicts is divided by two to correct the total.
    """
    conflicts = 0
    for y in range(gridSize):
        for x in range(gridSize):
            color = color_assignment[y, x]
            if color == -1:
                continue  # Ignore cells without a color assignment
            for yi, xi in get_adjacent_cells(y, x):
                if color_assignment[yi, xi] == color:
                    conflicts += 1
    # Since each conflict is counted twice, halve the result to get the true number
    return conflicts // 2 
###################################################################################################################################################

def get_conflicting_cells(color_assignment):
    """
    Identifies cells that are in conflict with their neighbors.
    Inputs:
    color_assignment: A 2D NumPy array representing the color of each cell.
    Outputs:
    conflicting_cells: A list of tuples representing the coordinates of conflicting cells.
    Purpose:
    To find cells that need to be reassigned a color to resolve conflicts.
    How it works:
    It goes through each cell in the grid.
    If a cell is already colored (i.e., it'isa pre-colored cell), it skips to the next cell.
    If the cell is uncolored (color is -1), it also skips that cell.
    For colored cells, it checks all adjacent cells.
    If any neighboring cell has the same color, the current cell is added to the set of conflicting cells.
    The use of a set ensures that each conflicting cell is only counted once, even if it has multiple conflicts.
    """
    conflicting_cells = set()
    for y in range(gridSize):
        for x in range(gridSize):
            if grid[y, x] != -1: # Skip if the cell is pre-colored
                continue 
            color = color_assignment[y, x]
            if color == -1:
                continue  # Skip unassigned cells
            for yi, xi in get_adjacent_cells(y, x):
                if color_assignment[yi, xi] == color:
                    conflicting_cells.add((y, x))
                    break  # No need to check further neighbors
    return list(conflicting_cells)
###################################################################################################################################################

def simulated_annealing(max_colors):
    """
    Performs the Simulated Annealing algorithm to find a valid color assignment.
    Inputs:
    max_colors: An integer representing the maximum number of colors to use.
    Outputs:
    color_assignment: A 2D NumPy array with the final color assignment.
    current_conflicts: An integer representing the number of conflicts in the final assignment.
    Purpose:
    To find a coloring of the grid that satisfies the no-adjacent-same-color constraint.
    How it works:
    Starts with a random color assignment.
    Iteratively selects a conflicting cell and tries assigning it a different color.
    Accepts new based on the change in conflicts and a probability that decreases over time (temperature).
    Stops when no conflicts remain or the temperature reaches a minimum threshold.
    """
    # change by monitoring the output.
    initial_temperature = 1000
    final_temperature = 0.1
    alpha = 0.99  # Cooling rate
    current_temperature = initial_temperature
    max_iterations = 100000

    # Start with a random initial solution
    color_assignment = np.full((gridSize, gridSize), -1, dtype=int)
    for y in range(gridSize):
        for x in range(gridSize):
            if grid[y, x] != -1:
                color_assignment[y, x] = grid[y, x]  # Keep pre-colored cells
            else:
                color_assignment[y, x] = random.randint(0, max_colors - 1)
    current_conflicts = calculate_conflicts(color_assignment)

    iteration = 0
    while current_temperature > final_temperature and current_conflicts > 0 and iteration < max_iterations:
        iteration += 1

        # Get conflicting cells
        conflicting_cells = get_conflicting_cells(color_assignment)
        if not conflicting_cells:
            break  # No conflicts

        # Randomly select a conflicting cell
        y, x = random.choice(conflicting_cells)

        # Generate a neighbor by changing the color of the conflicting cell
        new_color_assignment = color_assignment.copy()
        current_color = new_color_assignment[y, x]
        valid_colors = [color for color in range(max_colors) if color != current_color]
        if not valid_colors:
            continue  # No other colors to choose from
        new_color = random.choice(valid_colors)
        new_color_assignment[y, x] = new_color

        new_conflicts = calculate_conflicts(new_color_assignment)
        delta = new_conflicts - current_conflicts

        # Decide whether to accept the new state
        if delta < 0 or random.uniform(0, 1) < math.exp(-delta / current_temperature):
            color_assignment = new_color_assignment
            current_conflicts = new_conflicts

        # Decrease the temperature
        current_temperature *= alpha

        # Early exit if conflicts are zero
        if current_conflicts == 0:
            break

    return color_assignment, current_conflicts

# Attempt to color the grid using Simulated Annealing with increasing number of colors
max_colors = 4  # Start with 4 colors
while True:
    color_assignment, conflicts = simulated_annealing(max_colors)
    if conflicts == 0:
        break
    else:
        print(f"Conflicts remaining with {max_colors} colors: {conflicts}")
        if max_colors >= max_colors_available:
            print("Unable to find a conflict-free coloring with the given number of colors.")
            break
    max_colors += 1  # Increase the number of colors and try again
    
###################################################################################################################################################

def find_largest_shape_at(y, x):
    """
    Finds the largest possible shape that can be placed at a given position.
    Inputs:
    y: The row index of the starting cell.
    x: The column index of the starting cell.
    Outputs:
    Shape_index: An integer representing the index of the largest shape that fits.
                 Returns None if no shape fits.
    Purpose:
    To maximize the area covered by each shape placement, reducing the total number of shapes used.
    How it works:
    Generates a list of shape indices sorted by size in descending order.
    Iterates over the shapes to find the largest one that fits at the position without overlapping or mismatching colors.
    Checks grid boundaries and whether the shape overlaps with pre-colored cells or cells with different assigned colors.
    """
    shape_indices = list(range(len(shapes)))
    # Sort shapes by size (number of cells they cover), largest first
    shape_indices.sort(key=lambda idx: -np.sum(shapes[idx]))
    assigned_color = color_assignment[y, x]
    for shape_index in shape_indices:
        shape = shapes[shape_index]
        shape_height, shape_width = shape.shape
        # Check if shape fits within grid boundaries
        if x + shape_width > gridSize or y + shape_height > gridSize:
            continue
        can_place = True
        for i in range(shape_height):
            for j in range(shape_width):
                if shape[i, j]:
                    yi = y + i
                    xi = x + j
                    # Check if the cell is empty and matches the assigned color
                    if grid[yi, xi] != -1 or color_assignment[yi, xi] != assigned_color:
                        can_place = False
                        break
            if not can_place:
                break
        if can_place:
            return shape_index
    return None
##################################################################################################################################################
def place_shapes():
    """
    Places shapes on the grid according to the color assignment.
    Outputs:
    The function updates the grid and placedShapes through 'execute' commands.
    Purpose:
    To fill the grid efficiently with shapes that match the color assignment and cover as much area as possible.
    How it works:
    - Iterates over each cell in the grid.
    - Skips cells that are already visited, pre-colored, or unassigned.
    - Finds the largest shape that fits at the current position.
    - Sets the brush shape and color, moves the brush, and attempts to place the shape.
    - Marks cells as visited and updates the color assignment if placement is successful.
    """
    
    # here I will use above written functions
    
    global shapePos, currentShapeIndex, currentColorIndex, grid, placedShapes
    visited = set()
    for y in range(gridSize):
        for x in range(gridSize):
            # Skip if cell is already visited, not empty, or has no assigned color
            if (y, x) in visited or grid[y, x] != -1 or color_assignment[y, x] == -1:
                continue
            assigned_color = color_assignment[y, x]
            shape_index = find_largest_shape_at(y, x)
            if shape_index is not None:
                set_shape(shape_index)
                set_color(assigned_color)
                move_brush_to([x, y])
                # Check if placement was successful
                pre_grid = grid.copy()
                execute('place')
                shapePos, currentShapeIndex, currentColorIndex, grid, placedShapes, _ = execute('export')
                grid = np.array(grid, dtype=int)
                if np.array_equal(pre_grid, grid):
                    # Placement failed
                    continue
                # Mark cells covered by the shape as visited
                shape = shapes[shape_index]
                shape_height, shape_width = shape.shape
                for i in range(shape_height):
                    for j in range(shape_width):
                        if shape[i, j]:
                            yi, xi = y + i, x + j
                            visited.add((yi, xi))
                            color_assignment[yi, xi] = assigned_color  # Update color_assignment
            else:
                # Place a single cell shape if no larger shape fits
                set_shape(0)
                set_color(assigned_color)
                move_brush_to([x, y])
                pre_grid = grid.copy()
                execute('place')
                shapePos, currentShapeIndex, currentColorIndex, grid, placedShapes, _ = execute('export')
                grid = np.array(grid, dtype=int)
                if np.array_equal(pre_grid, grid):
                    # Placement failed
                    continue
                visited.add((y, x))
                color_assignment[y, x] = assigned_color  # Update color_assignment

# Call the function to place shapes on the grid
place_shapes()
# Update the variables after placing shapes
shapePos, currentShapeIndex, currentColorIndex, grid, placedShapes, _ = execute('export')
grid = np.array(grid, dtype=int)  # Ensure grid is an integer NumPy array

# just calculation to check overall performance of the algo
total_cells = gridSize * gridSize
total_colored_cells = np.count_nonzero(grid != -1)
cells_colored_by_agent = total_colored_cells - pre_colored_cells
unfilled_cells = total_cells - total_colored_cells


shapes_used = len(placedShapes)

# print(f"Shapes used: {shapes_used}")
# print(f"Unfilled cells: {unfilled_cells}")
# print(f"Cells colored by agent: {cells_colored_by_agent}")

###################################################################################################################################################

# Final conflict check to ensure no adjacent cells have the same color
def calculate_conflicts_in_grid(grid):
    """
    Calculates the number of conflicts in the final grid.
    Inputs:
    grid: A 2D NumPy array representing the final state of the grid.
    Outputs:
    conflicts: An integer representing the total number of conflicts.
    Purpose:
    To verify that the coloring constraints are satisfied in the final grid.
    How it works:
    - Iterates over each cell and checks its adjacent cells.
    - Increments the conflict count if an adjacent cell has the same color.
    - Divides the total conflicts by 2 to avoid double-counting.
    """
    conflicts = 0
    for y in range(gridSize):
        for x in range(gridSize):
            color = grid[y, x]
            if color == -1:
                continue  # Skip empty cells
            for yi, xi in get_adjacent_cells(y, x):
                if grid[yi, xi] == color:
                    conflicts += 1
    return conflicts // 2  # Each conflict is counted twice

# final_conflicts = calculate_conflicts_in_grid(grid)
# if final_conflicts == 0:
#     print("Final grid satisfies the coloring constraints.")
#     done = True  # Update done to True
# else:
#     print(f"Final grid has {final_conflicts} conflicts.")
#     done = False
    
########################################

# Do not modify any of the code below.

########################################

end = time.time()

np.savetxt('grid.txt', grid, fmt="%d")
with open("shapes.txt", "w") as outfile:
    outfile.write(str(placedShapes))
with open("time.txt", "w") as outfile:
    outfile.write(str(end - start))


#########################################################################################################################
# Acknowledgment
"""
While the core implementation and problem-solving approach in this code are my own work, I utilized large language models as a supportive tool throughout the development process.
Particularly, I have made use of large language models to get a deeper conceptual understanding of simulated annealings These tools have helped in debugging-find the logical errors.Most especially for the shape placement feature, I
used LLMs to just give me an idea of potential improvements which I integrated and modified into this particular problem. 
The LLM i used for the  clear comments for the more complex functions in order to improve the readability of the code.
Lastly,I  used to divide this big problem into subtasks which then drove how I implemented the solution. 
Actual coding, selection of algorithms, and solving the core problem has been done by myself.
"""
