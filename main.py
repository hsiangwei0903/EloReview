import os
import json
from role.simulator import SimulationManager

# ============================================
# Configuration: Set your API Key here
# ============================================
GEMINI_API_KEY = "" # Replace with your actual API key
# ============================================

def load_papers_from_file(filepath: str):
    """
    Load papers from a JSON file.

    Args:
        filepath: Path to JSON file containing paper list

    Returns:
        List of paper dictionaries
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_sample_papers():
    """
    Create a sample paper list for testing.
    You should replace this with your actual paper data.

    Returns:
        List of paper dictionaries with id, url, and actual_rating
    """
    papers = [
        {
            'id': 'paper_001',
            'url': 'https://openreview.net/forum?id={paper_ID}',
            'actual_rating': 7.5
        },
        {
            'id': 'paper_002',
            'url': 'https://openreview.net/forum?id={paper_ID}',
            'actual_rating': 6.8
        },
        {
            'id': 'paper_003',
            'url': 'https://openreview.net/forum?id={paper_ID}',
            'actual_rating': 8.2
        },
        {
            'id': 'paper_004',
            'url': 'https://openreview.net/forum?id={paper_ID}',
            'actual_rating': 5.5
        },
        {
            'id': 'paper_005',
            'url': 'https://openreview.net/forum?id={paper_ID}',
            'actual_rating': 7.0
        },
        {
            'id': 'paper_006',
            'url': 'https://openreview.net/forum?id={paper_ID}',
            'actual_rating': 6.2
        },
        {
            'id': 'paper_007',
            'url': 'https://openreview.net/forum?id={paper_ID}',
            'actual_rating': 8.5
        },
        {
            'id': 'paper_008',
            'url': 'https://openreview.net/forum?id={paper_ID}',
            'actual_rating': 7.8
        },
        {
            'id': 'paper_009',
            'url': 'https://openreview.net/forum?id={paper_ID}',
            'actual_rating': 6.5
        },
        {
            'id': 'paper_010',
            'url': 'https://openreview.net/forum?id={paper_ID}',
            'actual_rating': 7.2
        }
    ]
    return papers

def main():
    """
    Main function to run all three experiments.
    """
    print("=" * 60)
    print("ICLR Paper Review Simulation - Multi-Experiment Runner")
    print("=" * 60)

    # Step 1: Get API Key
    # Priority: 1. Config variable, 2. Environment variable
    api_key = GEMINI_API_KEY if GEMINI_API_KEY != "YOUR_API_KEY_HERE" else os.environ.get('GEMINI_API_KEY')

    if not api_key or api_key == "YOUR_API_KEY_HERE":
        print("\nError: GEMINI_API_KEY not set.")
        print("Please set your API key in one of these ways:")
        print("  1. Edit the GEMINI_API_KEY variable at the top of main.py")
        print("  2. Set environment variable: export GEMINI_API_KEY='your_api_key'")
        return

    print(f"\nAPI Key loaded: {api_key[:10]}...{api_key[-4:]}")

    # Step 2: Load or create papers
    paper_list_file = 'papers.json'

    if os.path.exists(paper_list_file):
        print(f"\nLoading papers from {paper_list_file}...")
        paper_list = load_papers_from_file(paper_list_file)
    else:
        print(f"\nNo {paper_list_file} found. Using sample papers...")
        paper_list = create_sample_papers()

        # Save sample papers to file for future use
        with open(paper_list_file, 'w', encoding='utf-8') as f:
            json.dump(paper_list, f, indent=2)
        print(f"Sample papers saved to {paper_list_file}")

    print(f"Loaded {len(paper_list)} papers")

    # Step 3: Configure simulation parameters
    num_rounds = 30
    random_seed = 42

    print(f"\nSimulation Configuration:")
    print(f"  - Number of rounds: {num_rounds}")
    print(f"  - Papers per round: 2")
    print(f"  - Reviewers per paper: 3")
    print(f"  - Total reviewers: 6 (bluffer, critic, expert, harmonizer, optimist, skimmer)")
    print(f"  - Random seed: {random_seed}")
    print(f"  - Experiments to run: 1, 2, 3")

    # Step 4: Create SimulationManager
    print("\nInitializing SimulationManager...")
    manager = SimulationManager(
        api_key=api_key,
        paper_list=paper_list,
        num_rounds=num_rounds,
        seed=random_seed
    )

    # Step 5: Run all three experiments
    print("\n" + "=" * 60)
    print("Starting All Experiments")
    print("=" * 60)
    print("\nExperiment Modes:")
    print("  Mode 1: Baseline (AC doesn't see Elo, reviewers don't update memory)")
    print("  Mode 2: AC sees Elo scores")
    print("  Mode 3: AC sees Elo + Reviewers update memory")
    print()

    try:
        manager.run_all_experiments()

        print("\n" + "=" * 60)
        print("All Experiments Completed Successfully!")
        print("=" * 60)
        print(f"\nResults saved to: simulation_results.json")
        print(f"Total results recorded: {len(manager.results)}")

    except Exception as e:
        print(f"\n❌ Error during simulation: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 6: Print summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)

    for exp_mode in [1, 2, 3]:
        exp_results = [r for r in manager.results if r['experiment'] == exp_mode]
        print(f"\nExperiment {exp_mode}:")
        print(f"  - Total papers reviewed: {len(exp_results)}")
        if exp_results:
            final_elos = exp_results[-1]['elos_after_round']
            print(f"  - Final Elo scores:")
            for r_id, elo in sorted(final_elos.items()):
                persona = manager.persona_names[r_id - 1]
                print(f"    Reviewer {r_id} ({persona}): {elo:.2f}")

if __name__ == "__main__":
    main()
