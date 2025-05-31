import random
import math
import numpy as np
from collections import Counter
import csv

# --- Configuration Constants ---
GAMES_IN_SEASON = 17  # Standard NFL games for PPG calculation
PICKS_PER_ROUND = 12  # Assumed number of teams/picks per round for ADP to Round conversion
DEFAULT_CSV_FILENAME = "player_pool.csv" # Default CSV filename

# --- 0. Define Player Data and Roster Constraints ---

def load_player_pool_from_csv(filename=DEFAULT_CSV_FILENAME):
    """
    Loads player data from a CSV file.
    Converts 'TotalPoints' to PPG and 'ADP' to a calculated draft round.
    Expected CSV format: ID,Name,Position,TotalPoints,ADP (and other optional columns)
    """
    player_pool_list = []
    # Define the headers your script expects for essential calculations
    expected_headers = ["ID", "Name", "Position", "TotalPoints", "ADP"]
    try:
        with open(filename, mode='r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            # Check if all expected headers are present
            if not reader.fieldnames or not all(field in reader.fieldnames for field in expected_headers):
                missing_headers = [h for h in expected_headers if h not in (reader.fieldnames or [])]
                print(f"Error: CSV file '{filename}' is missing required headers: {', '.join(missing_headers)} (or CSV is empty).")
                print(f"Expected headers are: {', '.join(expected_headers)}")
                return []

            for row_num, row in enumerate(reader, 1):
                try:
                    player_id = int(row["ID"])
                    name = row["Name"]
                    position = row["Position"]
                    
                    # Convert TotalPoints to PPG
                    total_points = float(row["TotalPoints"])
                    ppg = total_points / GAMES_IN_SEASON
                    
                    # Convert ADP to calculated draft round
                    adp = int(row["ADP"])
                    # ADP 1-12 -> Round 1, ADP 13-24 -> Round 2, etc.
                    calculated_round = math.ceil(adp / PICKS_PER_ROUND)
                    calculated_round = max(1, calculated_round) # Ensure round is at least 1

                    player_pool_list.append((player_id, name, position, ppg, calculated_round))
                except ValueError as e:
                    print(f"Warning: Skipping row {row_num} in {filename} due to data conversion error: {e}. Row content: {row}")
                except KeyError as e:
                    # This error means a column like 'TotalPoints' or 'ADP' was not found in a row,
                    # which should ideally be caught by the header check earlier, but good for safety.
                    print(f"Warning: Skipping row {row_num} in {filename} due to missing column: {e}. Row content: {row}")
                except ZeroDivisionError:
                    print(f"Warning: Skipping row {row_num} in {filename} due to ZeroDivisionError (GAMES_IN_SEASON or PICKS_PER_ROUND might be 0). Row: {row}")


    except FileNotFoundError:
        print(f"Error: Player pool CSV file '{filename}' not found.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while reading '{filename}': {e}")
        return []

    if not player_pool_list:
        print(f"Warning: No players loaded from '{filename}'. The CSV might be empty, headers incorrect, or all rows had errors.")
    return player_pool_list

# Load the player pool from CSV
PLAYER_POOL = load_player_pool_from_csv()

# If PLAYER_POOL is empty after trying to load, exit or handle error
if not PLAYER_POOL:
    print("Critical Error: Player pool is empty. Exiting.")
    exit() 

# The rest of the setup depends on PLAYER_POOL being populated
PLAYERS_BY_POSITION = {
    "QB": [p for p in PLAYER_POOL if p[2] == "QB"],
    "RB": [p for p in PLAYER_POOL if p[2] == "RB"],
    "WR": [p for p in PLAYER_POOL if p[2] == "WR"],
    "TE": [p for p in PLAYER_POOL if p[2] == "TE"]# Assuming DST is a position in your CSV
}

# Ensure players within each position are sorted by their calculated round
for pos in PLAYERS_BY_POSITION:
    PLAYERS_BY_POSITION[pos].sort(key=lambda x: x[4]) # x[4] is 'calculated_round'

ROSTER_STRUCTURE = {"QB": 2, "RB": 3, "WR": 5, "TE": 2} # Adjust if DST is not used
TOTAL_ROSTER_SPOTS = sum(ROSTER_STRUCTURE.values())

# --- 1. Genetic Algorithm Parameters ---
POPULATION_SIZE = 150
N_GENERATIONS = 150
MUTATION_RATE = 0.20
CROSSOVER_RATE = 0.8
TOURNAMENT_SIZE = 7
PENALTY_VIOLATION = 10000 # Penalty for violating constraints (e.g., duplicate rounds)

# --- Helper Functions ---
# Create a mapping from player ID to player data for quick lookups
PLAYER_ID_TO_DATA = {p[0]: p for p in PLAYER_POOL} # p[0] is player_id

def get_player_data(player_id):
    """Retrieves player data (tuple) by player ID."""
    return PLAYER_ID_TO_DATA.get(player_id)

def get_player_round(player_id):
    """Retrieves the calculated draft round of a player by ID."""
    player_data = get_player_data(player_id)
    # player_data[4] is 'calculated_round' from the tuple (id, name, pos, ppg, round)
    return player_data[4] if player_data else -1 

# --- 2. Chromosome Representation & Initialization ---
def create_individual():
    """
    Creates a single individual (roster) ensuring players are chosen according
    to ROSTER_STRUCTURE and attempting to adhere to unique player and unique round constraints.
    """
    individual_ids = [None] * TOTAL_ROSTER_SPOTS
    used_player_ids = set()
    used_rounds = set() # Tracks rounds used within this individual
    
    # Determine the position for each slot in the roster
    position_order = []
    for pos, count in ROSTER_STRUCTURE.items():
        for _ in range(count):
            position_order.append(pos)

    for i in range(TOTAL_ROSTER_SPOTS):
        position_to_fill = position_order[i]
        
        if not PLAYERS_BY_POSITION.get(position_to_fill):
            # This case should be rare if PLAYER_POOL is loaded correctly and ROSTER_STRUCTURE is valid
            # print(f"Warning: No players available for position {position_to_fill} in create_individual.")
            individual_ids[i] = -99 # Mark as highly invalid if a position has no available players
            continue

        # Attempt to find a player for the position with a unique ID and unique round
        possible_players = [
            p for p in PLAYERS_BY_POSITION[position_to_fill]
            if p[0] not in used_player_ids and p[4] not in used_rounds # p[0] is ID, p[4] is round
        ]
        
        if not possible_players:
            # If no player with a unique round is available, try for unique ID only
            possible_players = [p for p in PLAYERS_BY_POSITION[position_to_fill] if p[0] not in used_player_ids]
            
            if not possible_players:
                # Fallback: if all players of this position are used (should be rare with repair),
                # or if no players for this position (should be caught above).
                # Pick any player of this position. Repair function will handle duplicates.
                if PLAYERS_BY_POSITION[position_to_fill]:
                    chosen_player = random.choice(PLAYERS_BY_POSITION[position_to_fill])
                else: # Should not be reached if initial check passed
                    individual_ids[i] = -1 # Mark as invalid, needs repair
                    continue
            else:
                chosen_player = random.choice(possible_players)
        else:
            chosen_player = random.choice(possible_players)

        individual_ids[i] = chosen_player[0] # Store player ID
        used_player_ids.add(chosen_player[0])
        used_rounds.add(chosen_player[4]) # Add this player's round to used_rounds

    # Final check for any None or -1 slots (e.g. if a position had no players at all)
    for i in range(len(individual_ids)):
        if individual_ids[i] is None or individual_ids[i] == -1 or individual_ids[i] == -99:
            slot_pos_fallback = position_order[i]
            if PLAYERS_BY_POSITION.get(slot_pos_fallback) and PLAYERS_BY_POSITION[slot_pos_fallback]:
                # Last resort: pick any player for this position. Duplicates/round issues handled by repair.
                last_resort_player = random.choice(PLAYERS_BY_POSITION[slot_pos_fallback])
                individual_ids[i] = last_resort_player[0]
            else:
                individual_ids[i] = -99 # Mark as very invalid if position truly has no one
    return individual_ids

def create_initial_population():
    """Creates the initial population of individuals."""
    return [create_individual() for _ in range(POPULATION_SIZE)]

# --- 3. Fitness Function ---
def calculate_fitness(individual_ids):
    """
    Calculates the fitness of an individual.
    Fitness is based on total PPG, penalized for violations (duplicate players, duplicate rounds).
    Returns: fitness_score, total_ppg, set_of_rounds_used, list_of_player_data_tuples
    """
    total_ppg = 0
    
    # Basic validation of the individual structure
    if not individual_ids or len(individual_ids) != TOTAL_ROSTER_SPOTS or \
       any(pid is None or not isinstance(pid, int) or pid <= 0 for pid in individual_ids): # pid <=0 includes -1, -99
        return -float('inf'), 0, set(), [] # Highly unfit

    lineup_players_data = [] # List to store (id, name, pos, ppg, round) tuples for this lineup
    for pid in individual_ids:
        p_data = get_player_data(pid)
        if p_data is None: # Player ID not found in main pool
             return -float('inf'), 0, set(), [] # Highly unfit
        lineup_players_data.append(p_data)

    # Constraint 1: No duplicate players
    player_id_set = set(p_data[0] for p_data in lineup_players_data) # p_data[0] is ID
    if len(player_id_set) != TOTAL_ROSTER_SPOTS:
        # Penalize heavily for duplicate players
        return -PENALTY_VIOLATION * 5, 0, set(), lineup_players_data 

    # Constraint 2: No duplicate rounds (one player per draft round)
    player_rounds = [p_data[4] for p_data in lineup_players_data] # p_data[4] is 'calculated_round'
    round_counts = Counter(player_rounds)
    # Calculate penalty based on how many rounds are over-used
    round_violations = sum(count - 1 for count in round_counts.values() if count > 1)
    if round_violations > 0:
        return -PENALTY_VIOLATION * round_violations, 0, set(player_rounds), lineup_players_data

    # If all constraints met, calculate total PPG
    for p_data in lineup_players_data:
        total_ppg += p_data[3] # p_data[3] is 'ppg'
    
    return total_ppg, total_ppg, set(player_rounds), lineup_players_data


# --- Repair Function (Crucial) ---
def get_position_for_slot_index(index):
    """Maps a roster slot index to its designated position."""
    # This needs to align with ROSTER_STRUCTURE and its iteration order in create_individual
    # Example: QB (1), RB (2), WR (3), TE (1), DST (1) -> Total 8 slots
    # Slot 0: QB
    # Slot 1, 2: RB
    # Slot 3, 4, 5: WR
    # Slot 6: TE
    # Slot 7: DST 
    # This is a simple implementation; a more robust one would build this map from ROSTER_STRUCTURE
    # For now, using the example structure:
    if index == 0: return "QB"
    if 1 <= index <= 2: return "RB" # Assuming RB1, RB2
    if 3 <= index <= 5: return "WR" # Assuming WR1, WR2, WR3
    if index == 6: return "TE"
    if index == 7: return "DST" # If DST is the 8th player
    # Adjust this function if your ROSTER_STRUCTURE or TOTAL_ROSTER_SPOTS changes
    return None # Should not happen if index is within TOTAL_ROSTER_SPOTS

def repair_lineup(lineup_ids):
    """
    Attempts to repair an individual lineup to satisfy constraints
    (unique players, unique rounds).
    """
    if not lineup_ids or len(lineup_ids) != TOTAL_ROSTER_SPOTS or \
       any(pid is None or not isinstance(pid, int) or pid <= 0 for pid in lineup_ids):
        # If lineup is fundamentally broken, create a new one
        return create_individual()

    repaired_ids = list(lineup_ids) # Work on a copy

    # Pass 1: Fix duplicate player IDs and fill invalid slots (-1, -99)
    for _ in range(TOTAL_ROSTER_SPOTS * 2): # Iterate a few times to allow changes to propagate
        id_counts = Counter(pid for pid in repaired_ids if pid > 0) # Count valid positive IDs
        made_change_in_pass_players = False
        
        for i in range(len(repaired_ids)):
            player_id = repaired_ids[i]
            slot_pos = get_position_for_slot_index(i)
            if not slot_pos or not PLAYERS_BY_POSITION.get(slot_pos): continue # Skip if slot/pos invalid

            # If slot is invalid OR player is duplicated
            if player_id <= 0 or id_counts.get(player_id, 0) > 1:
                current_ids_in_lineup = set(pid for pid in repaired_ids if pid > 0 and pid != player_id)
                
                # Try to find a replacement that isn't already in the lineup
                replacement_options = [p for p in PLAYERS_BY_POSITION[slot_pos] if p[0] not in current_ids_in_lineup]
                
                if replacement_options:
                    new_player = random.choice(replacement_options)
                    if player_id > 0: id_counts[player_id] -= 1 # Decrement old player count if it was valid
                    repaired_ids[i] = new_player[0]
                    id_counts[new_player[0]] = id_counts.get(new_player[0], 0) + 1
                    made_change_in_pass_players = True
                else:
                    # If no unique player can be found for this slot (e.g., all players of this position are used)
                    # This is a tough spot. For now, mark it for later or accept a potential duplicate temporarily.
                    # The fitness function will heavily penalize this.
                    # Or, if the slot was invalid, try to pick any from the position.
                    if player_id <=0 and PLAYERS_BY_POSITION[slot_pos]:
                         repaired_ids[i] = random.choice(PLAYERS_BY_POSITION[slot_pos])[0]
                         made_change_in_pass_players = True


        if not made_change_in_pass_players: # If a full pass made no changes, player duplicates might be resolved
            break
    
    # Pass 2: Fix duplicate rounds
    for _ in range(TOTAL_ROSTER_SPOTS * 2): # Iterate a few times
        current_lineup_player_data_for_rounds = []
        valid_indices_for_round_repair = []
        current_ids_in_lineup_for_rounds = set()

        for idx, pid_val in enumerate(repaired_ids):
            if pid_val > 0:
                p_data = get_player_data(pid_val)
                if p_data:
                    current_lineup_player_data_for_rounds.append(p_data)
                    valid_indices_for_round_repair.append(idx)
                    current_ids_in_lineup_for_rounds.add(pid_val)
            else: # If any player ID is still invalid after player repair, this lineup is problematic
                return repaired_ids # Give up detailed round repair if player IDs are not even set

        if len(current_lineup_player_data_for_rounds) != TOTAL_ROSTER_SPOTS:
            # If not a full valid roster of players, round repair is premature / complex
            return repaired_ids 

        # Map: original_index_in_lineup -> player_round
        player_rounds_map = {idx: p_data[4] for idx, p_data in zip(valid_indices_for_round_repair, current_lineup_player_data_for_rounds)}
        round_counts = Counter(player_rounds_map.values())
        made_change_in_round_pass = False

        for original_idx_in_lineup in valid_indices_for_round_repair: # Iterate by original index
            current_round_of_player_at_idx = player_rounds_map.get(original_idx_in_lineup)

            if current_round_of_player_at_idx is None or round_counts[current_round_of_player_at_idx] <= 1:
                continue # This player's round is not duplicated, or player data is missing

            # This player's round is duplicated, try to change this player
            slot_pos = get_position_for_slot_index(original_idx_in_lineup)
            if not slot_pos or not PLAYERS_BY_POSITION.get(slot_pos): continue

            # Gather rounds and IDs of OTHER players in the lineup
            other_players_rounds = set()
            other_player_ids = set()
            for k_idx, k_pid in enumerate(repaired_ids):
                if k_idx == original_idx_in_lineup or k_pid <= 0: # Skip self or invalid
                    continue
                k_pdata = get_player_data(k_pid)
                if k_pdata:
                    other_players_rounds.add(k_pdata[4]) # k_pdata[4] is round
                    other_player_ids.add(k_pid)
            
            # Find a replacement for the current slot that has a unique ID and a unique round
            replacement_options = [
                p for p in PLAYERS_BY_POSITION[slot_pos]
                if p[0] not in other_player_ids and p[4] not in other_players_rounds # p[0] is ID, p[4] is round
            ]
            
            if replacement_options:
                new_player = random.choice(replacement_options)
                
                # Update counts and map for the current pass
                old_round = player_rounds_map.get(original_idx_in_lineup)
                if old_round is not None: round_counts[old_round] -= 1
                
                repaired_ids[original_idx_in_lineup] = new_player[0] # Update the ID in the lineup
                
                round_counts[new_player[4]] = round_counts.get(new_player[4], 0) + 1
                player_rounds_map[original_idx_in_lineup] = new_player[4] # Update the round in our map for this pass
                made_change_in_round_pass = True
        
        if not made_change_in_round_pass: # If a full pass made no changes to rounds
            break
            
    return repaired_ids


# --- 4. Selection Operator ---
def tournament_selection(population, fitness_scores_only):
    """Selects an individual using tournament selection."""
    # Ensure population is not empty and TOURNAMENT_SIZE is valid
    if not population or len(population) < TOURNAMENT_SIZE:
        # Fallback or error handling if population is too small
        return random.choice(population) if population else create_individual()

    selected_indices = random.sample(range(len(population)), TOURNAMENT_SIZE)
    
    tournament_individuals = [population[i] for i in selected_indices]
    tournament_fitnesses = [fitness_scores_only[i] for i in selected_indices]
    
    winner_index_in_tournament = np.argmax(tournament_fitnesses)
    return tournament_individuals[winner_index_in_tournament]

# --- 5. Crossover Operator ---
def crossover(parent1_ids, parent2_ids):
    """Performs crossover between two parents to produce two children."""
    child1_ids, child2_ids = list(parent1_ids), list(parent2_ids) # Make copies
    if random.random() < CROSSOVER_RATE:
        if len(parent1_ids) > 1 and len(parent1_ids) == TOTAL_ROSTER_SPOTS: # Ensure valid parents
            # Perform single-point crossover
            pt = random.randint(1, len(parent1_ids) - 1)
            child1_ids = parent1_ids[:pt] + parent2_ids[pt:]
            child2_ids = parent2_ids[:pt] + parent1_ids[pt:]
    
    # Repair children to ensure they meet constraints
    child1_ids = repair_lineup(child1_ids)
    child2_ids = repair_lineup(child2_ids)
    return child1_ids, child2_ids

# --- 6. Mutation Operator ---
def mutate(individual_ids):
    """Performs mutation on an individual."""
    mutated_ids = list(individual_ids) # Make a copy
    if random.random() < MUTATION_RATE:
        if not mutated_ids or len(mutated_ids) != TOTAL_ROSTER_SPOTS :
             return repair_lineup(mutated_ids) # Repair if malformed

        mutation_idx = random.randrange(len(mutated_ids)) # Index of player to mutate
        original_player_id = mutated_ids[mutation_idx]
        slot_pos = get_position_for_slot_index(mutation_idx)

        if not slot_pos or not PLAYERS_BY_POSITION.get(slot_pos):
            return repair_lineup(mutated_ids) # Repair if slot position is problematic

        # Gather rounds and IDs of OTHER players in the lineup
        other_players_rounds = set()
        other_player_ids = set()
        for i in range(len(mutated_ids)):
            if i == mutation_idx or mutated_ids[i] <= 0: # Skip self or invalid players
                continue
            p_data = get_player_data(mutated_ids[i])
            if p_data:
                other_players_rounds.add(p_data[4]) # p_data[4] is round
                other_player_ids.add(p_data[0])   # p_data[0] is ID

        original_player_round = get_player_round(original_player_id) if original_player_id > 0 else -1

        # Try to find a replacement that has a unique ID and unique round
        candidate_replacements = [
            p for p in PLAYERS_BY_POSITION[slot_pos]
            if p[0] not in other_player_ids and \
               (p[4] not in other_players_rounds or \
                (original_player_round != -1 and p[4] == original_player_round)) # Allow same round if it's the original mutated player's round
        ]
        
        if not candidate_replacements:
            # Fallback: unique ID, even if round is duplicated (repair/fitness will handle)
            candidate_replacements = [p for p in PLAYERS_BY_POSITION[slot_pos] if p[0] not in other_player_ids and p[0] != original_player_id]

        if candidate_replacements:
            new_player = random.choice(candidate_replacements)
            mutated_ids[mutation_idx] = new_player[0]
        # If no replacements found, the original player remains (or repair will handle if it was invalid)

    return repair_lineup(mutated_ids) # Always repair after mutation

# --- 7. Main Genetic Algorithm Loop ---
def genetic_algorithm_adp_lineup():
    if not PLAYER_POOL: 
        print("Critical: genetic_algorithm_adp_lineup called with empty PLAYER_POOL.")
        return None, -float('inf')

    population = create_initial_population()
    if not population:
        print("Critical: Initial population is empty. Check create_individual and player pool.")
        return None, -float('inf')
    print(f"Initial population of {len(population)} ADP-based lineups created.")

    best_lineup_overall_ids = None
    best_fitness_overall = -float('inf')
    best_points_overall = 0 # Stores the actual PPG sum for the best valid lineup

    for generation in range(N_GENERATIONS):
        # Calculate fitness for each individual in the population
        # fitness_results is list of (fitness_score, total_ppg, set_of_rounds, list_of_player_data)
        fitness_results = [calculate_fitness(ind) for ind in population]
        fitness_scores_only = [res[0] for res in fitness_results] # Just the raw fitness scores

        # Identify the best individual in the current generation
        current_best_idx_in_gen = np.argmax(fitness_scores_only)
        current_best_fitness_in_gen = fitness_scores_only[current_best_idx_in_gen]
        current_best_ppg_in_gen = fitness_results[current_best_idx_in_gen][1]


        # Update overall best if current generation's best is better
        if current_best_fitness_in_gen > best_fitness_overall:
            best_fitness_overall = current_best_fitness_in_gen
            best_lineup_overall_ids = list(population[current_best_idx_in_gen]) # Store a copy
            
            # If the new best is valid (not heavily penalized), store its actual points
            if best_fitness_overall > -PENALTY_VIOLATION: # Check if it's a valid lineup
                best_points_overall = current_best_ppg_in_gen
            else:
                best_points_overall = 0 # Reset if the "best" is actually an invalid penalized one

            points_display = f"{best_points_overall:.2f}" if best_fitness_overall > -PENALTY_VIOLATION else "N/A (invalid)"
            print(f"Gen {generation + 1}: New Global Best! Fitness={best_fitness_overall:.2f}, Actual PPG={points_display}")

        # Create the next generation
        next_population_ids = []
        
        # Elitism: Carry over a few of the best individuals from current generation
        elite_count = max(1, int(0.05 * POPULATION_SIZE)) # e.g., 5%
        # Get indices of individuals sorted by fitness (descending)
        sorted_indices_current_gen = np.argsort(fitness_scores_only)[::-1] 
        
        elites_added_this_gen = 0
        for i in range(len(sorted_indices_current_gen)):
            if elites_added_this_gen >= elite_count:
                break
            elite_candidate_idx = sorted_indices_current_gen[i]
            # Only add elites if they are reasonably valid (not extremely penalized)
            if fitness_scores_only[elite_candidate_idx] > -PENALTY_VIOLATION * 2: 
                 candidate_lineup_for_elite = list(population[elite_candidate_idx])
                 # Avoid adding exact duplicates to elites if already present
                 is_duplicate_elite = any(Counter(existing_elite) == Counter(candidate_lineup_for_elite) for existing_elite in next_population_ids)
                 if not is_duplicate_elite:
                    next_population_ids.append(candidate_lineup_for_elite)
                    elites_added_this_gen +=1
        
        # Fill the rest of the population using selection, crossover, and mutation
        while len(next_population_ids) < POPULATION_SIZE:
            parent1 = tournament_selection(population, fitness_scores_only)
            parent2 = tournament_selection(population, fitness_scores_only)
            child1, child2 = crossover(parent1, parent2)
            
            next_population_ids.append(mutate(child1))
            if len(next_population_ids) < POPULATION_SIZE:
                next_population_ids.append(mutate(child2))
        
        population = next_population_ids[:POPULATION_SIZE] # Ensure population size is maintained

        # Periodic print of generation's best
        if (generation + 1) % 10 == 0 or generation == N_GENERATIONS -1 :
            # Recalculate for the current state of the population before printing
            current_pop_fitness_results = [calculate_fitness(ind) for ind in population]
            current_pop_fitness_scores = [res[0] for res in current_pop_fitness_results]
            idx_best_in_current_pop = np.argmax(current_pop_fitness_scores)
            
            gen_fitness, gen_points, gen_rounds_set, _ = current_pop_fitness_results[idx_best_in_current_pop]
            points_display_periodic = f"{gen_points:.2f}" if gen_fitness > -PENALTY_VIOLATION else 'N/A (invalid)'
            rounds_display_periodic = len(gen_rounds_set) if gen_fitness > -PENALTY_VIOLATION else 'N/A (invalid)'
            print(f"Gen {generation + 1}: Current Pop Best Fitness={gen_fitness:.2f}, Actual PPG={points_display_periodic}, Unique Rounds Used={rounds_display_periodic}")

    print("\n--- Genetic Algorithm for ADP-Based Lineup Finished ---")
    if best_lineup_overall_ids and best_fitness_overall > -PENALTY_VIOLATION / 2 : # Check if a reasonably valid solution was found
        # Recalculate final stats for the absolute best lineup found
        final_fitness, final_points, final_rounds_set, final_player_data_list = calculate_fitness(best_lineup_overall_ids)
        
        print(f"Best Lineup Found (Player IDs): {best_lineup_overall_ids}")
        print(f"Stats: Final Fitness={final_fitness:.2f}, Total Actual PPG={final_points:.2f}, Unique Rounds Used={len(final_rounds_set)}")
        print("\nBest Lineup Details:")
        
        if final_player_data_list and len(final_player_data_list) == TOTAL_ROSTER_SPOTS:
            # Sort by original roster slot order for display (which get_position_for_slot_index implies)
            # This requires mapping original_player_ids to their data for consistent order if repair changed things.
            # For simplicity, we'll print based on the order in final_player_data_list, which comes from best_lineup_overall_ids
            
            # Create a temporary map from ID to data for the best lineup
            best_lineup_data_map = {p_data[0]: p_data for p_data in final_player_data_list}
            
            displayed_count = 0
            for i, player_id_in_slot in enumerate(best_lineup_overall_ids):
                p_data = best_lineup_data_map.get(player_id_in_slot)
                slot_pos_display = get_position_for_slot_index(i) # Get position for the slot
                if p_data:
                    # p_data is (id, name, position, ppg, round)
                    print(f"  - Slot {i} ({slot_pos_display or p_data[2]}): {p_data[1]} (ID: {p_data[0]}) - PPG: {p_data[3]:.2f}, Calc Round: {p_data[4]}")
                    displayed_count +=1
                else:
                    print(f"  - Slot {i} ({slot_pos_display}): Error - Player ID {player_id_in_slot} data not found in final list.")

            if displayed_count != TOTAL_ROSTER_SPOTS:
                 print("Warning: Not all player details could be displayed for the best lineup.")

            # Final validation checks on the best lineup
            if len(final_rounds_set) != TOTAL_ROSTER_SPOTS:
                print(f"WARNING: Final lineup does not use {TOTAL_ROSTER_SPOTS} unique rounds! (Used: {len(final_rounds_set)})")
            if len(set(p_data[0] for p_data in final_player_data_list)) != TOTAL_ROSTER_SPOTS:
                 print(f"WARNING: Final lineup contains DUPLICATE players (post-processing check)!")
            else:
                print("Best lineup appears valid based on final constraint checks.")
        else:
            print("Best lineup data is malformed or incomplete for detailed printout.")
        return best_lineup_overall_ids, best_fitness_overall
    else:
        print("No significantly valid solution found, or the solution did not improve enough beyond penalties.")
        if best_lineup_overall_ids: # If there was some "best" even if bad
             print(f"Last best attempt (may be highly penalized): {best_lineup_overall_ids} with fitness {best_fitness_overall:.2f}")
             # Try to print details of this penalized lineup
             _, _, _, last_attempt_data = calculate_fitness(best_lineup_overall_ids)
             if last_attempt_data:
                print("Details of last attempt:")
                for p_d in last_attempt_data:
                    print(f"  - {p_d[2]} {p_d[1]}: PPG={p_d[3]:.2f}, Round={p_d[4]}")

        else:
            print("No lineup was ever selected as best_lineup_overall_ids during the process.")
        return None, -float('inf')

# --- 8. Run the Genetic Algorithm ---
if __name__ == "__main__":
    # Ensure your CSV file (e.g., "pp2.csv") is in the same directory as the script,
    # or provide the full path to load_player_pool_from_csv.
    # The CSV should have columns: ID, Name, Position, TotalPoints, ADP
    
    print(f"Attempting to load player data from: {DEFAULT_CSV_FILENAME}")
    print(f"PPG will be calculated assuming {GAMES_IN_SEASON} games per season.")
    print(f"Draft rounds will be calculated assuming {PICKS_PER_ROUND} picks per round.")
    print("-" * 30)

    best_lineup_ids, best_fitness_val = genetic_algorithm_adp_lineup()
    
    print("\n" + "=" * 30)
    print("--- Main Program Output ---")
    if best_lineup_ids:
        print(f"Returned Best Lineup IDs: {best_lineup_ids}")
        print(f"Returned Best Fitness: {best_fitness_val:.2f}")
        # Further details about the best lineup are printed within genetic_algorithm_adp_lineup
    else:
        print("The genetic algorithm did not return a best lineup, or the lineup was invalid/heavily penalized.")
    print("=" * 30)

