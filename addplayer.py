import csv
import os

def get_player_info():
    """
    Prompts the user for player information and validates input.
    Returns a dictionary containing player data or None if input is invalid.
    """
    player = {}
    print("\nEnter player details:")

    # Get Player ID
    while True:
        try:
            player_id_str = input("Enter Player ID (must be an integer): ").strip()
            if not player_id_str:
                print("Player ID cannot be empty.")
                continue
            player['ID'] = int(player_id_str)
            break
        except ValueError:
            print("Invalid input. Player ID must be an integer.")

    # Get Player Name
    while True:
        player_name = input("Enter Player Name: ").strip()
        if not player_name:
            print("Player Name cannot be empty.")
        else:
            player['Name'] = player_name
            break

    # Get Player Position
    while True:
        player_pos = input("Enter Player Position (e.g., QB, RB, WR, TE, DST): ").strip().upper()
        if not player_pos:
            print("Player Position cannot be empty.")
        else:
            player['Position'] = player_pos
            break

    # Get Projected Points
    while True:
        try:
            points_str = input("Enter Projected Points (must be a number): ").strip()
            if not points_str:
                print("Projected Points cannot be empty.")
                continue
            player['Points'] = float(points_str)
            break
        except ValueError:
            print("Invalid input. Projected Points must be a number (e.g., 25.5).")

    # Get Assigned Draft Round
    while True:
        try:
            round_str = input("Enter Assigned Draft Round (must be an integer): ").strip()
            if not round_str:
                print("Assigned Draft Round cannot be empty.")
                continue
            player['Round'] = int(round_str)
            break
        except ValueError:
            print("Invalid input. Assigned Draft Round must be an integer.")

    return player

def main():
    """
    Main function to collect player data and write to CSV.
    """
    filename = "player_pool.csv"
    players_data = []
    fieldnames = ['ID', 'Name', 'Position', 'Points', 'Round']

    # Check if file exists to ask about overwriting or appending
    file_exists = os.path.isfile(filename)
    mode = 'w' # Default to write (overwrite)
    write_header = True

    if file_exists:
        while True:
            choice = input(f"File '{filename}' already exists. Do you want to (O)verwrite, (A)ppend, or (C)ancel? [O/A/C]: ").strip().upper()
            if choice == 'O':
                mode = 'w'
                write_header = True
                break
            elif choice == 'A':
                mode = 'a'
                write_header = False # Don't write header if appending to existing file
                # Check if existing file is empty or has no header, then write header
                if os.path.getsize(filename) == 0:
                    write_header = True
                else:
                    # Check if header actually exists
                    try:
                        with open(filename, 'r', newline='', encoding='utf-8') as temp_f:
                            reader = csv.reader(temp_f)
                            existing_header = next(reader, None)
                            if not existing_header or [h.strip() for h in existing_header] != fieldnames:
                                print(f"Warning: Existing file '{filename}' does not have the expected header or is malformed. Appending might lead to issues.")
                                # Optionally, force header if malformed, or handle differently
                    except Exception as e:
                        print(f"Could not read existing file header: {e}. Assuming header needs to be written if appending.")
                        write_header = True


                break
            elif choice == 'C':
                print("Operation cancelled.")
                return
            else:
                print("Invalid choice. Please enter O, A, or C.")

    print(f"\nStarting player data entry. Press Ctrl+C or leave Player ID empty at prompt to stop early (though not fully implemented here for empty ID stop).")
    print("Type 'done' (without quotes) for Player ID when you are finished adding players.")

    while True:
        print("-" * 20)
        # Modified prompt to allow 'done' to exit
        id_input_check = input("Enter Player ID (or type 'done' to finish): ").strip()
        if id_input_check.lower() == 'done':
            break
        
        # Re-do the ID input with validation if not 'done'
        player = {}
        while True:
            try:
                if not id_input_check: # If user just pressed Enter after the initial 'done' check prompt
                    id_input_check = input("Player ID cannot be empty. Enter Player ID (or type 'done' to finish): ").strip()
                    if id_input_check.lower() == 'done':
                        break # Break inner loop
                if id_input_check.lower() == 'done': # Check again if 'done' was entered in re-prompt
                    break # Break inner loop
                player['ID'] = int(id_input_check)
                break # Break inner loop (ID is valid number)
            except ValueError:
                id_input_check = input("Invalid input. Player ID must be an integer (or type 'done' to finish): ").strip()
        
        if id_input_check.lower() == 'done': # Break outer loop if 'done'
            break

        # Get Player Name

        # Get Player Position
        while True:
            player_pos = input("Enter Player Position (e.g., QB, RB, WR, TE, DST): ").strip().upper()
            if not player_pos:
                print("Player Position cannot be empty.")
            else:
                player['Position'] = player_pos
                break
        
        while True:
            player_name = input("Enter Player Name: ").strip()
            if not player_name:
                print("Player Name cannot be empty.")
            else:
                player['Name'] = player['Position']+'_'+player_name
                break

        # Get Projected Points
        while True:
            try:
                points_str = input("Enter Projected Points (must be a number): ").strip()
                if not points_str:
                    print("Projected Points cannot be empty.")
                    continue
                player['Points'] = float(points_str)
                break
            except ValueError:
                print("Invalid input. Projected Points must be a number (e.g., 25.5).")

        # Get Assigned Draft Round
        while True:
            try:
                round_str = input("Enter Assigned Draft Round (must be an integer): ").strip()
                if not round_str:
                    print("Assigned Draft Round cannot be empty.")
                    continue
                player['Round'] = int(round_str)
                break
            except ValueError:
                print("Invalid input. Assigned Draft Round must be an integer.")
        
        players_data.append(player)
        print(f"Player '{player['Name']}' added.")

    if not players_data:
        print("\nNo players were added. CSV file will not be modified or created.")
        return

    try:
        with open(filename, mode=mode, newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(players_data)
        print(f"\nPlayer data successfully saved to '{filename}'")
    except IOError:
        print(f"Error: Could not write to file '{filename}'. Check permissions or path.")
    except Exception as e:
        print(f"An unexpected error occurred while writing the CSV: {e}")

if __name__ == "__main__":
    main()
