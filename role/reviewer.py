import os
import re
import httpx 
import io 
import json
from typing import Dict, Any, Optional
from google import genai 
from google.genai.errors import APIError
from .utils import REVIEW_SCHEMA, clean_and_load_json

class LLMAgentReviewer:
    """
    Represents an individual LLM agent reviewer for the ICLR simulation.
    Handles two-stage review generation, memory integration, and Elo updates.
    """
    def __init__(self, reviewer_id: int, persona: str, client: genai.Client, initial_elo: float = 1500.0):
        self.reviewer_id: int = reviewer_id
        self.persona: str = persona
        self.elo_score: float = initial_elo
        self.memory_prompt: str = "" 
        self._base_system_prompt: str = self._load_persona_prompt()
        self.review_policy_prompt: str = self._load_review_policy_prompt()
        self.llm_client: genai.Client = client 
        self.llm_model: str = 'gemini-2.5-flash'
        self.REVIEW_SCHEMA = REVIEW_SCHEMA

    def _load_persona_prompt(self) -> str:
        file_path = f"prompt/persona/{self.persona}.txt"
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _load_review_policy_prompt(self) -> str:
        file_path = "prompt/reviewer.txt"
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _get_current_system_prompt(self) -> str:
        """
        Constructs the full system prompt, including the base persona and the memory prompt.
        """
        full_prompt = self._base_system_prompt
        
        if self.memory_prompt:
            memory_section = "\n\n--- Reviewer Self-Reflection/Memory ---\n"
            memory_section += "Reference the following self-correction instructions to generate your review, but **DO NOT** change your fundamental persona defined above:\n"
            memory_section += self.memory_prompt
            memory_section += "\n----------------------------------------"
            full_prompt += memory_section
            
        return full_prompt
    
    def _parse_rating(self, review_text: str) -> Optional[int]:
        """
        Extracts the numerical rating (0, 2, 4, 6, 8, 10) from the review text using regex.
        """
        match = re.search(r"\bscore[:\s]+([0246810])\b", review_text)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None 
        return None

    def _upload_pdf_from_url(self, paper_pdf_url: str):
        """
        Fetches the PDF content from the URL and uploads it to the Gemini API.
        """
        print(f"-> Reviewer {self.reviewer_id} fetching and uploading PDF from: {paper_pdf_url}")
        try:
            # 1. Fetch content using httpx
            response = httpx.get(paper_pdf_url)
            response.raise_for_status() 
            
            # 2. Convert content to a file-like object
            doc_io = io.BytesIO(response.content)
            
            # 3. Upload the file using the Gemini client
            uploaded_file = self.llm_client.files.upload(
                file=doc_io,
                config=dict(mime_type='application/pdf')
            )
            print(f"-> PDF uploaded successfully. File Name: {uploaded_file.name}")
            return uploaded_file
        except httpx.HTTPError as e:
            print(f"Error fetching PDF from URL: {e}")
            return None
        except APIError as e:
            print(f"Error uploading file to Gemini API: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during PDF upload: {e}")
            return None

    def _cleanup_file(self, uploaded_file):
        """
        Deletes the uploaded file from the Gemini API service to manage resources.
        """
        if uploaded_file:
            try:
                self.llm_client.files.delete(name=uploaded_file.name)
                print(f"-> Cleaned up uploaded file: {uploaded_file.name}")
            except APIError as e:
                print(f"Warning: Failed to delete uploaded file {uploaded_file.name}: {e}")

    def generate_review_stage_1(self, paper_pdf_url: str) -> Dict[str, Any]:
        """
        Stage 1: Generates an initial review and rating without seeing other reviews.
        """
        uploaded_file = None
 
        uploaded_file = self._upload_pdf_from_url(paper_pdf_url)
        if not uploaded_file:
            return {"initial_rating": None, "initial_review_text": "Error: Failed to process PDF.", "system_prompt_used": ""}

        # 2. Construct the prompt
        full_system_prompt = self._get_current_system_prompt()
        
        user_prompt = (
            f"You are conducting the **first stage** of the review process. "
            f"Review the paper provided as context. "
            f"Generate your initial review following the required policy.\n\n"
            f"Review Policy:\n{self.review_policy_prompt}"
        )

        print(f"-> Reviewer {self.reviewer_id} calling Gemini API for Stage 1...")
        
        # --- STRUCTURED OUTPUT CONFIGURATION ---
        generation_config = genai.types.GenerateContentConfig(
            system_instruction=full_system_prompt,
            response_mime_type="application/json",
            response_schema=self.REVIEW_SCHEMA
        )

        response = self.llm_client.models.generate_content(
            model=self.llm_model,
            contents=[uploaded_file, user_prompt], 
            config=generation_config
        )

        # 4. Process JSON response directly
        # The response.text should already be clean JSON if the API succeeds
        review_data = json.loads(response.text)
        initial_rating = review_data.get("score", None)

        return {
            "initial_rating": initial_rating,
            "initial_review_text": response.text, # Store the full JSON string
            "system_prompt_used": full_system_prompt
        }


    def generate_review_stage_2(self, paper_pdf_url: str, initial_review: Dict[str, Any], other_reviews: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Stage 2: Generates an adjusted review and final rating after seeing others.
        """
        uploaded_file = None

        # 1. Upload the PDF file
        uploaded_file = self._upload_pdf_from_url(paper_pdf_url)
        if not uploaded_file:
            return {"final_rating": None, "final_review_text": "Error: Failed to process PDF.", "system_prompt_used": ""}
        
        # 2. Format other reviews for the prompt
        formatted_other_reviews = "\n\n--- Other Reviewers' Feedback ---\n"
        for r_id, review_data in other_reviews.items():
            formatted_other_reviews += (
                # NOTE: Here we assume initial_review_text (the Stage 1 output) 
                # is the full JSON string, so we just pass the score.
                f"\nReviewer {r_id} (Score: {review_data['initial_rating']}):\n"
                f"Review Details: {review_data['initial_review_text']}\n" 
            )
        formatted_other_reviews += "\n----------------------------------------\n"
        
        # 3. Construct the prompt
        full_system_prompt = self._get_current_system_prompt()
        
        user_prompt = (
            f"You are conducting the **second stage** of the review process (Discussion/Rebuttal Phase). "
            f"Your initial review and rating were: Score {initial_review['initial_rating']}.\n"
            f"Your Initial Review (JSON):\n{initial_review['initial_review_text']}\n\n"
            f"Review the paper provided as context, in light of the other reviews.\n"
            f"You have the **option to adjust your rating**, but you must ensure your second stage review remains **ALIGNED with your core persona** (defined in the system prompt).\n"
            f"Critically re-evaluate your rating and adjust your review text if necessary. "
            f"Generate your final review following the required JSON policy.\n"
            f"{formatted_other_reviews}\n\n"
            f"Review Policy:\n{self.review_policy_prompt}"
        )

        print(f"-> Reviewer {self.reviewer_id} calling Gemini API for Stage 2 (Adjustment)...")

        # --- STRUCTURED OUTPUT CONFIGURATION ---
        generation_config = genai.types.GenerateContentConfig(
            system_instruction=full_system_prompt,
            response_mime_type="application/json",
            response_schema=self.REVIEW_SCHEMA
        )

        # 4. Call the Gemini API
        response = self.llm_client.models.generate_content(
            model=self.llm_model,
            contents=[uploaded_file, user_prompt], 
            config=generation_config
        )

        # 5. Process response
        review_data = json.loads(response.text)
        final_rating = review_data.get("score", None)

        return {
            "final_rating": final_rating,
            "final_review_text": response.text, # Store the full JSON string
            "system_prompt_used": full_system_prompt
        }

    def update_elo(self, elo_adjustment: float):
        """
        Updates the reviewer's Elo score by applying a pre-calculated adjustment.
        
        Args:
            elo_adjustment (float): The amount to change the Elo score by (e.g., +100.0, -50.0).
        
        Returns:
            tuple[float, float]: The Elo score before and after the update.
        """
        old_elo = self.elo_score
        
        # Apply the direct adjustment
        self.elo_score += elo_adjustment
        
        print(f"Reviewer {self.reviewer_id} Elo Updated: Old Elo={old_elo:.2f}, New Elo={self.elo_score:.2f} (Change: {elo_adjustment:+.2f})")
        
        # Return the required values for the subsequent memory update function
        return old_elo, self.elo_score

    def update_memory(self, elo_before: float, elo_after: float, previous_review_text: str, previous_review_score: int):
        """
        The reviewer agent brainstorms a new strategy to maintain or increase its Elo score
        without changing its core persona, and updates its memory prompt.

        Args:
            elo_before (float): Elo score before the update.
            elo_after (float): Elo score after the update.
            previous_review_text (str): The final review text from the last round.
            previous_review_score (int): The AC's quality score for the last review.
        """
        elo_change = elo_after - elo_before
        old_memory_prompt = self.memory_prompt # Capture the current memory before overwrite
        
        # 1. Construct the Brainstorming Prompt
        strategy_system_prompt = (
            "You are a strategic AI agent whose sole goal is to **optimize the reviewer's Elo score** "
            "by adjusting their tactical approach, NOT their core persona or values. "
            "Your output must be a single, concise set of instructions (5 to 10 sentences) "
            "that will be prepended to the reviewer's system prompt for the next round. "
            "This instruction must help the reviewer perform a higher-quality review without contradicting their persona."
        )

        user_prompt = (
            f"--- Performance Analysis for Reviewer {self.reviewer_id} ---\n"
            f"**Core Persona:**\n{self._base_system_prompt}\n\n"
            f"**Previous Tactical Instruction (Memory Used Last Round):**\n"
            f"'{old_memory_prompt}'\n\n"
            f"**Performance Metrics:**\n"
            f"Elo Score Change: {elo_change:+.2f} (Previous Elo: {elo_before:.2f} -> Current Elo: {elo_after:.2f})\n"
            f"**Last Review:**\n{previous_review_text}\n\n"
        )
        
        if elo_change < -5:
            user_prompt += "Your Elo score dropped significantly. Your new strategy MUST address this failure."
        elif elo_change > 5:
            user_prompt += "Your Elo score increased. Refine your successful strategy to ensure continued high performance."
        else:
            user_prompt += "Your Elo score remained stable. Identify minor improvements to push your score higher."
            
        user_prompt += (
            "\n\nBased on the above, brainstorm a **new tactical instruction** for the next round. "
            "This instruction must preserve the Core Persona but maintain or improve your Elo score to prevent penalty."
            "\n\n**New Tactical Instruction (Single paragraph output):**"
        )
        
        # 2. Call the Gemini API to generate the new memory prompt (No try/except)
        print(f"-> Reviewer {self.reviewer_id} brainstorming new strategy...")
        
        response = self.llm_client.models.generate_content(
            model=self.llm_model,
            contents=[user_prompt],
            config=genai.types.GenerateContentConfig(
                system_instruction=strategy_system_prompt
            )
        )

        new_strategy = response.text.strip()
        
        # 3. Update the memory prompt
        self.memory_prompt = new_strategy
        print(f"Reviewer {self.reviewer_id} memory updated. New strategy: '{new_strategy}'")

if __name__ == '__main__':
    # --- 0. Configuration and Setup ---
    API_KEY = os.environ.get('GEMINI_API_KEY')
    gemini_client = genai.Client(api_key=API_KEY)

    # Paper to review
    PAPER_URL = "https://openreview.net/pdf?id=6TLdqAZgzn"
    # Placeholder AC-assigned quality score for testing update_elo/update_memory
    AC_QUALITY_SCORE = 3
    
    # --- 1. Setup Required Dummy Files ---
    # Since the agent loads prompts from files, we must create placeholders.
    BASE_DIR = './'
    PROMPT_DIR = os.path.join(BASE_DIR, 'prompt')
    PERSONA_DIR = os.path.join(PROMPT_DIR, 'persona')
    
    os.makedirs(PERSONA_DIR, exist_ok=True)
    
    print("--- Starting LLMAgentReviewer Test ---")
    
    # --- 2. Initialize Reviewers ---
    reviewer1 = LLMAgentReviewer(reviewer_id=101, persona='expert', client=gemini_client, initial_elo=1500.0)
    reviewer2 = LLMAgentReviewer(reviewer_id=102, persona='critic', client=gemini_client, initial_elo=1500.0)
    
    print(f"\nReviewer 2 Initial Elo: {reviewer2.elo_score:.2f}")
    # --- 3. Simulate Stage 1 Review for Reviewer 1 and 2 ---
    stage1_r1 = reviewer1.generate_review_stage_1(PAPER_URL)
    print("\n## Stage 1: Initial Review (R1)")
    print(f"R1 Initial Rating: {stage1_r1.get('initial_rating')}")
    print(f"R1 Review Text Snippet: {stage1_r1.get('initial_review_text', '')[:150]}...")
    print("\n## Stage 1: Initial Review (R2)")
    stage1_r2 = reviewer2.generate_review_stage_1(PAPER_URL)
    print(f"R2 Initial Rating: {stage1_r2.get('initial_rating')}")
    print(f"R2 Review Text Snippet: {stage1_r2.get('initial_review_text', '')[:150]}...")

    print(f"Reviewer 2 Initial Elo: {reviewer2.elo_score:.2f}")
    
    # --- 5. Simulate Stage 2 Review (Adjustment) for Reviewer 2 ---
    print("\n## Stage 2: Adjusted Review (R2)")

    other_reviews = {
        reviewer2.reviewer_id: {
            "initial_rating": stage1_r1.get('initial_rating'),
            "initial_review_text": stage1_r1.get('initial_review_text')
        }
    }
    
    stage2_r2 = reviewer2.generate_review_stage_2(PAPER_URL, stage1_r2, other_reviews)
    print(f"R2 Final Rating: {stage2_r2.get('final_rating')}")
    print(f"R2 Final Review Snippet: {stage2_r2.get('final_review_text', '')[:150]}...")

    # --- 6. Simulate Elo Update and Memory Brainstorming ---
    print("\n## Elo and Memory Update")
    
    # a. Update Elo using the placeholder AC quality score
    elo_before, elo_after = reviewer2.update_elo(AC_QUALITY_SCORE)
    
    # b. Update Memory (Brainstorming)
    reviewer2.update_memory(
        elo_before=elo_before, 
        elo_after=elo_after, 
        previous_review_text=stage2_r2['final_review_text'], 
        previous_review_score=AC_QUALITY_SCORE
    )

    # --- 7. Final Status Check ---
    print("\n## Final Status")
    print(f"R2 Final Elo: {reviewer2.elo_score:.2f}")
    print(f"R2 Memory Prompt (Next Round): '{reviewer2.memory_prompt}'")
    
    # Test memory injection (just print the full system prompt for the next round)
    print("\n--- Next Round System Prompt Preview ---")
    print(reviewer2._get_current_system_prompt())
    
    print("\n--- Test Complete ---")