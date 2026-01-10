import os
import json
import random
import numpy as np
from google import genai
from typing import Dict, List, Any, Optional, Tuple
from .reviewer import LLMAgentReviewer
from .area_chair import AreaChairAgent

class SimulationManager:
    """
    Manages the multi-round simulation of ICLR paper review using LLM agents,
    implementing a rank-based Elo system (+100, 0, -100) and three distinct 
    experimental modes.
    """

    def __init__(self, api_key: str, paper_list: List[Dict[str, Any]], num_rounds: int = 5, seed: Optional[int] = 42):
        """
        Initializes the Simulation Manager.

        Args:
            api_key (str): The Google AI API key.
            paper_list (List[Dict]): List of papers with ID and actual_rating.
            num_rounds (int): The number of rounds (review cycles) to run.
            seed (Optional[int]): Random seed for reproducibility. Default is 42.
        """
        # 1. Setup API Client
        os.environ['GEMINI_API_KEY'] = api_key
        # Assuming genai.Client is initialized elsewhere or handles API key internally
        self.client = genai.Client(api_key=api_key) 
        
        # 2. Simulation Parameters
        self.num_rounds: int = num_rounds
        self.persona_names: List[str] = ['bluffer', 'critic', 'expert', 'harmonizer', 'optimist', 'skimmer']
        self.num_reviewers_per_paper: int = 3
        self.num_papers_per_round: int = 2
        
        # --- Random Seed Setup ---
        self.seed = seed
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
        # -------------------------
        
        # 3. Agents and Data Storage
        self.reviewers: Dict[int, 'LLMAgentReviewer'] = {}
        # NOTE: Assuming AreaChairAgent is accessible
        self.ac_agent: 'AreaChairAgent' = AreaChairAgent(client=self.client) 
        self.papers: List[Dict[str, Any]] = paper_list
        self.results: List[Dict[str, Any]] = []

    def _initialize_reviewers(self, initial_elo: float = 1500.0):
        """
        Initializes the 6 LLM reviewer agents with unique IDs and personas, and resets memory.
        """
        print("--- Initializing Reviewer Agents ---")
        self.reviewers = {}
        for i, persona_name in enumerate(self.persona_names):
            reviewer_id = i + 1
            # NOTE: Assuming LLMAgentReviewer is accessible and uses 'persona' arg
            self.reviewers[reviewer_id] = LLMAgentReviewer(
                reviewer_id=reviewer_id,
                persona=persona_name,
                client=self.client,
                initial_elo=initial_elo
            )
            print(f"Initialized Reviewer {reviewer_id}: {persona_name} (Elo: {initial_elo})")
        
        # Reset memory for all reviewers when initializing
        for reviewer in self.reviewers.values():
            reviewer.memory_prompt = ""

    def _get_reviewer_elos(self) -> Dict[int, float]:
        """Returns a dictionary of current reviewer IDs and their Elo scores."""
        return {r_id: agent.elo_score for r_id, agent in self.reviewers.items()}

    def _calculate_ranking_rewards(self, reviewer_evaluations: List[Dict[str, Any]]) -> Dict[int, float]:
        """
        Calculates Elo adjustment based on rank (1st=+100, 2nd=0, 3rd=-100), 
        handling ties by distributing rewards.
        
        Args:
            reviewer_evaluations (List): List of AC evaluation dicts (must contain 'score' and 'id').
            
        Returns:
            Dict[int, float]: Map of Reviewer ID to their Elo adjustment.
        """
        # List of (score, reviewer_id) tuples
        scores_and_ids = []
        for item in reviewer_evaluations:
            # item['id'] is like "R01", item['score'] is an integer (e.g., 8)
            r_id = int(item['id'].replace('R', ''))
            scores_and_ids.append((item['score'], r_id))
        
        # Sort by score in descending order
        scores_and_ids.sort(key=lambda x: x[0], reverse=True)
        
        # Define base rewards for ranks 1, 2, 3 (Total reward pool = 0)
        base_rewards = [100.0, 0.0, -100.0]
        rewards: Dict[int, float] = {}

        i = 0
        while i < len(scores_and_ids):
            current_score = scores_and_ids[i][0]
            j = i # Start of the current tie group
            
            # Find the end of the current tie group
            while j < len(scores_and_ids) and scores_and_ids[j][0] == current_score:
                j += 1
            
            # Calculate the total reward pool for this tied group
            reward_pool = sum(base_rewards[i:j])
            group_size = j - i
            
            # Distribute the reward equally
            adjustment = reward_pool / group_size
            
            # Assign the calculated adjustment to all members of the group
            for k in range(i, j):
                reviewer_id = scores_and_ids[k][1]
                rewards[reviewer_id] = adjustment
            
            i = j # Move to the next rank group
            
        return rewards

    def _run_single_round(self, round_num: int, available_papers: List[Dict[str, Any]], all_reviewer_ids: List[int], experiment_mode: int):
        """
        Executes one full round of review for all assigned papers, implementing 
        random triplet assignment, rank-based Elo, and memory logging.
        """
        print(f"\n======== Running Round {round_num} | Experiment {experiment_mode} ========")
        
        ac_sees_elo = (experiment_mode in [2, 3])
        reviewer_updates_memory = (experiment_mode == 3)
        round_results = []

        # --- Paper Assignment (Random Triplet Switching) ---
        
        # Select N papers for this round
        papers_to_review_indices = random.sample(range(len(available_papers)), self.num_papers_per_round)
        
        paper_assignment: Dict[int, List[int]] = {}
        for paper_index in papers_to_review_indices:
            # Randomly select 3 unique reviewers (different triplet each time)
            assigned_reviewers = random.sample(all_reviewer_ids, self.num_reviewers_per_paper)
            paper_assignment[paper_index] = assigned_reviewers

        # --- Review Loop ---
        for paper_index in paper_assignment:
            reviewer_ids = paper_assignment[paper_index]
            paper = available_papers[paper_index]
            paper_id = paper['id']
            # Assume URL needs formatting if it uses {paper_ID}
            paper_url = paper['url'].format(paper_ID=paper_id) if '{paper_ID}' in paper['url'] else paper['url']
            paper_actual_rating = paper['actual_rating']
            
            print(f"\n--- Paper {paper_id} assigned to Reviewers {reviewer_ids} ---")

            # --- Stage 1: Initial Review ---
            stage1_reviews: Dict[int, Dict[str, Any]] = {}
            for r_id in reviewer_ids:
                stage1_reviews[r_id] = self.reviewers[r_id].generate_review_stage_1(paper_url)

            # --- Stage 2: Adjusted Review ---
            stage2_reviews: Dict[int, Dict[str, Any]] = {}
            for r_id in reviewer_ids:
                reviewer = self.reviewers[r_id]
                other_reviews = {k: v for k, v in stage1_reviews.items() if k != r_id}
                stage2_reviews[r_id] = reviewer.generate_review_stage_2(
                    paper_url, 
                    stage1_reviews[r_id], 
                    other_reviews
                )
            
            # --- AC Decision ---
            current_elos = self._get_reviewer_elos()
            ac_decision = self.ac_agent.make_final_decision(
                paper_id, 
                stage2_reviews, 
                current_elos, 
                ac_sees_elo
            )
            print(f"-> AC Decision: {ac_decision.get('final_decision')}")

            # --- Elo Update and Memory Update ---
            
            # 1. Calculate Rank-Based Adjustments
            review_evaluations = ac_decision.get('review_evaluation', [])
            elo_adjustments = self._calculate_ranking_rewards(review_evaluations) # {r_id: adjustment}
            
            # Prepare Reviewer Data for Recording and Update
            reviews_data_for_paper = {}
            for r_id in reviewer_ids:
                reviewer = self.reviewers[r_id]
                
                # 2. Apply Elo Change
                elo_adjustment = elo_adjustments.get(r_id, 0.0)
                elo_before, elo_after = reviewer.update_elo(elo_adjustment)
                
                # 3. Update Memory (Experiment 3 only)
                review_quality_score = None
                ac_evaluation = next((item for item in review_evaluations if item['id'] == f"R{r_id:02d}"), None)
                if ac_evaluation:
                     review_quality_score = ac_evaluation['score']
                
                if reviewer_updates_memory and review_quality_score is not None:
                    reviewer.update_memory(
                        elo_before=elo_before, 
                        elo_after=elo_after, 
                        previous_review_text=stage2_reviews[r_id]['final_review_text'], 
                        previous_review_score=review_quality_score
                    )

                # 4. Record Results, including memory prompt
                review_details = {**stage1_reviews[r_id], **stage2_reviews[r_id]}
                review_details['final_memory_prompt_used'] = reviewer.memory_prompt # Capture memory
                
                reviews_data_for_paper[r_id] = review_details

            # --- Record Results ---
            round_results.append({
                'experiment': experiment_mode,
                'round': round_num,
                'paper_id': paper_id,
                'actual_rating': paper_actual_rating,
                'ac_decision': ac_decision,
                'reviews': reviews_data_for_paper,
                'elos_after_round': self._get_reviewer_elos()
            })

        self.results.extend(round_results)
        print(f"======== Round {round_num} Completed | Experiment {experiment_mode} ========")
        
    
    def run_experiment(self, experiment_mode: int):
        """
        Runs the full multi-round simulation for a specific experiment mode.
        """
        print(f"\n\n==============================================")
        print(f"🚀 STARTING EXPERIMENT {experiment_mode}")
        print(f"==============================================")

        # Reset random seed at the start of each experiment to ensure same papers
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            print(f"Random seed reset to {self.seed} for experiment {experiment_mode}")

        self._initialize_reviewers()
        all_reviewer_ids = list(self.reviewers.keys())

        for round_num in range(1, self.num_rounds + 1):
            # Pass the necessary data to the single round function
            self._run_single_round(round_num, self.papers, all_reviewer_ids, experiment_mode)

        print(f"\n==============================================")
        print(f"✅ EXPERIMENT {experiment_mode} COMPLETED")
        print(f"==============================================")


    def run_all_experiments(self):
        """Runs experiments 1, 2, and 3 sequentially."""
        for mode in [1, 2, 3]:
            self.run_experiment(mode)

        # Output the results to a file for analysis
        with open('simulation_results.json', 'w') as f:
            json.dump(self.results, f, indent=4)
        print(f"\nAll experiments complete. Results saved to 'simulation_results.json'.")