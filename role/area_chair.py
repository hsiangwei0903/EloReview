import os
import json
from typing import Dict, Any, List, Optional

from google import genai
from google.genai.errors import APIError

from .utils import clean_and_load_json, AC_RESPONSE_SCHEMA

class AreaChairAgent:
    """
    Represents the Area Chair (AC) responsible for making final decisions 
    based on reviewer input and optionally Elo scores.
    """

    def __init__(self, client: genai.Client):
        """
        Initializes the AC Agent.
        """
        self.llm_client = client
        self.llm_model = 'gemini-2.5-flash'
        self.system_prompt = self._load_ac_prompt()
        self.review_schema = AC_RESPONSE_SCHEMA # Use the fixed schema
        
    def _load_ac_prompt(self) -> str:
        """
        Loads the AC system prompt from prompt/ac.txt.
        """
        # Note: Keeping the try/except here for file loading robustness, as it's outside the main simulation loop.
        try:
            file_path = "prompt/ac.txt"
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            print(f"Error: AC prompt file not found at {file_path}. Please create it.")
            return (
                "You are an Area Chair (AC) responsible for making final decisions on papers. "
                "Evaluate all provided reviewer submissions, including their final ratings. "
                "Generate your output in the requested JSON format."
            )

    def _prepare_reviewer_data(self, final_reviews: Dict[int, Dict[str, Any]], reviewer_elos: Dict[int, float], disclose_elo: bool) -> str:
        """
        Formats all reviewer data (reviews, ratings, and optionally Elo scores) 
        into a structured string for the LLM.
        """
        data_lines = []
        for r_id, review_data in final_reviews.items():
            elo = reviewer_elos.get(r_id, 1500.0)
            short_id = f"R{r_id:02d}" 
            
            review_block = f"--- Reviewer {short_id} ---\n"
            # NOTE: final_rating is already an integer from LLMAgentReviewer's post-processing
            review_block += f"Final Rating: {review_data.get('final_rating', 'N/A')}\n"
            
            if disclose_elo:
                review_block += f"Reviewer Elo Score: {elo:.2f}\n"
            
            review_block += "Review Text (JSON):\n"
            # Pass the full JSON text of the review for AC evaluation
            review_block += f"{review_data.get('final_review_text', 'No text provided.')}\n" 
            review_block += "-------------------------"
            data_lines.append(review_block)
            
        return "\n".join(data_lines)

    def make_final_decision(self, paper_title: str, final_reviews: Dict[int, Dict[str, Any]], reviewer_elos: Dict[int, float], disclose_elo: bool) -> Dict[str, Any]:
        """
        Calls the LLM to make the final Accept/Reject decision and evaluate the reviews.

        Args:
            paper_title (str): The title of the paper being reviewed.
            final_reviews (Dict): The final (Stage 2) reviews and ratings from all reviewers.
            reviewer_elos (Dict): The current Elo scores of all reviewers.
            disclose_elo (bool): Whether to provide the AC with the reviewer Elo scores.

        Returns:
            Dict[str, Any]: The parsed and processed JSON decision from the AC agent.
        """
        # 1. Prepare data for the prompt
        reviewer_data_string = self._prepare_reviewer_data(final_reviews, reviewer_elos, disclose_elo)
        
        # 2. Construct the user prompt
        elo_context = "The reviewers' Elo scores are INCLUDED for your consideration." if disclose_elo else "The reviewers' Elo scores are NOT provided."

        user_prompt = (
            f"Paper Title: {paper_title}\n\n"
            f"Task: Make a final decision (Accept/Reject) on this paper. {elo_context}\n"
            f"You must also evaluate the quality of each individual review on a scale of 0 to 10 (even numbers only).\n\n"
            f"--- All Review Data ---\n"
            f"{reviewer_data_string}\n"
            f"----------------------------------------"
        )
        
        # 3. Configure the API call for JSON output
        generation_config = genai.types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=self.review_schema,
            system_instruction=self.system_prompt
        )

        print(f"-> AC Agent generating final decision for '{paper_title}'...")

        # 4. Call the Gemini API (No try/except as requested)
        response = self.llm_client.models.generate_content(
            model=self.llm_model,
            contents=[user_prompt],
            config=generation_config
        )
        
        # 5. Parse the JSON text response
        json_text = response.text
        decision_data = json.loads(json_text)
        
        # 6. CRITICAL FIX: Convert string scores back to integers for Python logic (Elo update)
        for eval_item in decision_data.get('review_evaluation', []):
            # Convert the score from string (e.g., "8") to integer (8)
            eval_item['score'] = int(eval_item['score'])
        
        # Basic validation of the decision key (useful for debugging simulation issues)
        if decision_data.get('final_decision') not in ["Accept", "Reject"]:
            raise ValueError(f"AC agent returned an invalid final_decision value: {decision_data.get('final_decision')}")

        return decision_data