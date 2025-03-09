import sys
sys.path.insert(0, '../bomberman')
import subprocess
import os
from events import Event

# Available worlds
worlds = {
    "variant1": "variant1.py",
    "variant2": "variant2.py", 
    "variant3": "variant3.py",
    "variant4": "variant4.py",
    "variant5": "variant5.py"
}

def train_on_world(variant_file, num_games=5):
    """Train agent on a specific world variant"""
    stats = {
        'wins': 0,
        'monster_deaths': 0,
        'explosion_deaths': 0,
        'timeouts': 0,
        'total_score': 0
    }
    
    for i in range(num_games):
        print(f"Run iteration {i+1}")
        try:
            # Run the variant file with the proper Python environment
            env = os.environ.copy()
            env['PYTHONPATH'] = '../bomberman'  # Set Python path
            
            # Set working directory to where the variant files and map are
            variant_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "project2")
            
            # Run the game and wait for completion
            result = subprocess.run(
                ["python3", variant_file],
                env=env,
                capture_output=True,
                text=True,
                cwd=variant_dir  # Run from project2 directory where map.txt is
            )
            
            # Check if the process actually ran
            if result.returncode != 0:
                print(f"Game failed with error: {result.stderr}")
                continue
            
            # Parse the output to determine game result
            output = result.stdout
            if "me found the exit" in output:
                stats['wins'] += 1
                print(f"Game {i}: Won!")
            elif "self" in output and not "selfp" in output:
                stats['explosion_deaths'] += 1
                print(f"Game {i}: Killed by bomb")
            elif "killed" in output:
                stats['monster_deaths'] += 1
                print(f"Game {i}: Killed by monster")
            else:
                stats['timeouts'] += 1
                print(f"Game {i}: Timeout/Other")
                print(f"Output: {output}")  # Debug output
            
            # Print progress every 10 games
            if (i + 1) % 10 == 0:
                print(f"\nProgress after {i+1} games:")
                print(f"Win rate: {stats['wins']/(i+1):.2%}")
                print(f"Deaths by monster: {stats['monster_deaths']/(i+1):.2%}")
                print(f"Deaths by explosion: {stats['explosion_deaths']/(i+1):.2%}")
                print(f"Timeouts: {stats['timeouts']/(i+1):.2%}\n")
                
        except Exception as e:
            print(f"Game {i} failed with error: {e}")
            continue
            
    return stats

def main():
    num_rounds = 10  # Number of rounds to train (you can modify this)
    
    print(f"Training for {num_rounds} rounds on {len(worlds)} worlds")
    
    # Store stats for each variant and overall
    variant_stats = {world: {
        'wins': 0,
        'monster_deaths': 0,
        'explosion_deaths': 0,
        'timeouts': 0,
        'total_score': 0
    } for world in worlds.keys()}
    
    total_stats = {
        'wins': 0,
        'monster_deaths': 0,
        'explosion_deaths': 0,
        'timeouts': 0,
        'total_score': 0
    }
    
    # Train for specified number of rounds
    for round_num in range(num_rounds):
        print(f"\n=== Starting Round {round_num + 1}/{num_rounds} ===")
        
        # Train on each world in order
        for world_name, variant_file in worlds.items():
            print(f"\nTraining on {world_name}")
            stats = train_on_world(variant_file, num_games=1)  # One game per variant per round
            
            # Accumulate statistics for this variant
            for key in stats:
                variant_stats[world_name][key] += stats[key]
                total_stats[key] += stats[key]
        
        # Print progress after each round
        print(f"\n--- Round {round_num + 1} Complete ---")
        print(f"Current Win Rate: {total_stats['wins']/((round_num + 1) * len(worlds)):.2%}")
    
    # Print final results for each variant
    print("\n=== Final Results ===")
    print("\nResults by Variant:")
    total_games_per_variant = num_rounds
    for world_name, stats in variant_stats.items():
        print(f"\n{world_name}:")
        print(f"Wins: {stats['wins']}/{total_games_per_variant} ({stats['wins']/total_games_per_variant:.2%})")
        print(f"Deaths by monster: {stats['monster_deaths']}/{total_games_per_variant} ({stats['monster_deaths']/total_games_per_variant:.2%})")
        print(f"Deaths by explosion: {stats['explosion_deaths']}/{total_games_per_variant} ({stats['explosion_deaths']/total_games_per_variant:.2%})")
        print(f"Timeouts: {stats['timeouts']}/{total_games_per_variant} ({stats['timeouts']/total_games_per_variant:.2%})")
    
    total_games = num_rounds * len(worlds)
    print("\nOverall Results:")
    print(f"Total Wins: {total_stats['wins']}/{total_games} ({total_stats['wins']/total_games:.2%})")
    print(f"Total Deaths by monster: {total_stats['monster_deaths']}/{total_games} ({total_stats['monster_deaths']/total_games:.2%})")
    print(f"Total Deaths by explosion: {total_stats['explosion_deaths']}/{total_games} ({total_stats['explosion_deaths']/total_games:.2%})")
    print(f"Total Timeouts: {total_stats['timeouts']}/{total_games} ({total_stats['timeouts']/total_games:.2%})")

if __name__ == '__main__':
    main() 