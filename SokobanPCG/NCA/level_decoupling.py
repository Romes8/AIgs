import os

def parse_sokoban_file(input_file, output_dir):
    """Parse a file containing multiple Sokoban levels and save them as individual files."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    current_level = []
    level_number = 0
    
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            
            # If we find a level marker or empty line and we have collected a level
            if (line.startswith(';') or not line) and current_level:
                # Save the current level
                save_level(current_level, level_number, output_dir)
                current_level = []
                level_number += 1
            # If it's not a level marker and not an empty line, it's part of a level
            elif not line.startswith(';') and line:
                current_level.append(line)
    
    # Save the last level if there is one
    if current_level:
        save_level(current_level, level_number, output_dir)
        level_number += 1
    
    print(f"Successfully processed {level_number} levels")

def save_level(level_lines, level_number, output_dir):
    """Save a single level to a file."""
    filename = f'level_{level_number:04d}.txt'
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        f.write('\n'.join(level_lines))
    
    # Print first level as example
    if level_number == 0:
        print("\nExample level (first level):")
        print('\n'.join(level_lines))

if __name__ == "__main__":
    input_file = 'sokoban_levels.txt'  # Your input file name
    output_dir = 'data/sokoban_levels/'
    
    print("Starting level processing...")
    parse_sokoban_file(input_file, output_dir)