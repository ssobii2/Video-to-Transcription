import os
import time
import asyncio
from typing import Optional, Callable
import logging

from openai import AsyncOpenAI

from config import Config

logger = logging.getLogger(__name__)

class AIService:
    """AI service for processing transcriptions with OpenAI"""
    
    def __init__(self, config: Config):
        self.config = config
        self.client = None
        
        if config.openai_api_key:
            self.client = AsyncOpenAI(api_key=config.openai_api_key)
        else:
            logger.warning("OpenAI API key not provided - AI features will be disabled")
    
    async def process_transcription(
        self,
        transcription_text: str,
        custom_prompt: str = "",
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """
        Process transcription with AI using custom prompt
        
        Args:
            transcription_text: The transcription to process
            custom_prompt: Custom prompt for processing
            progress_callback: Optional callback for progress updates
            
        Returns:
            AI-processed response text
        """
        if not self.client:
            raise ValueError("OpenAI API key not configured")
        
        if not transcription_text.strip():
            raise ValueError("Transcription text is empty")
        
        # Prepare the prompt
        if custom_prompt.strip():
            system_prompt = f"""You are a helpful assistant that processes transcriptions based on user instructions.
            
User instruction: {custom_prompt}

Please process the following transcription according to the user's instruction:"""
        else:
            system_prompt = """You are a helpful assistant that summarizes transcriptions into clear, organized points.
            
Please summarize the following transcription into clear, well-organized bullet points:"""
        
        if progress_callback:
            await progress_callback("Sending to AI for processing...")
        
        try:
            start_time = time.time()
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": transcription_text}
            ]
            
            response = await self.client.chat.completions.create(
                model=self.config.openai_model,
                messages=messages,
                max_tokens=4000,
                temperature=0.7,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            
            processing_time = time.time() - start_time
            
            if progress_callback:
                await progress_callback(f"AI processing completed in {processing_time:.1f} seconds")
            
            ai_response = response.choices[0].message.content
            
            if not ai_response:
                raise ValueError("Empty response from AI")
            
            logger.info(f"AI processing completed in {processing_time:.1f} seconds")
            return ai_response.strip()
            
        except Exception as e:
            logger.error(f"AI processing failed: {e}")
            if progress_callback:
                await progress_callback(f"AI processing failed: {str(e)}")
            raise
    
    async def save_ai_response(
        self,
        ai_response: str,
        original_filename: str,
        custom_prompt: str = ""
    ) -> str:
        """
        Save AI response to file
        
        Args:
            ai_response: The AI response text
            original_filename: Original audio/video filename
            custom_prompt: The prompt used for processing
            
        Returns:
            Path to saved AI response file
        """
        # Generate filename for AI response
        base_name = os.path.splitext(original_filename)[0]
        ai_filename = f"{base_name}_ai_response.txt"
        ai_path = os.path.join(self.config.ai_responses_folder, ai_filename)
        
        try:
            with open(ai_path, 'w', encoding='utf-8') as f:
                f.write(f"AI Response for: {original_filename}\n")
                f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Model: {self.config.openai_model}\n")
                if custom_prompt:
                    f.write(f"Prompt: {custom_prompt}\n")
                f.write("-" * 50 + "\n\n")
                f.write(ai_response)
            
            logger.info(f"AI response saved: {ai_filename}")
            return ai_path
            
        except Exception as e:
            logger.error(f"Failed to save AI response: {e}")
            # Clean up partial file
            if os.path.exists(ai_path):
                os.remove(ai_path)
            raise
    
    def is_available(self) -> bool:
        """Check if AI service is available"""
        return self.client is not None
    
    def get_model_info(self) -> dict:
        """Get information about AI model configuration"""
        return {
            "available": self.is_available(),
            "model": self.config.openai_model if self.is_available() else None,
            "api_key_configured": bool(self.config.openai_api_key)
        }
    
    async def test_connection(self) -> bool:
        """Test AI service connection"""
        if not self.client:
            return False
        
        try:
            # Simple test request
            response = await self.client.chat.completions.create(
                model=self.config.openai_model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            return True
        except Exception as e:
            logger.error(f"AI service connection test failed: {e}")
            return False
    
    def get_suggested_prompts(self) -> list:
        """Get list of suggested prompts for transcription processing"""
        return [
            {
                "name": "Summarize in Points",
                "prompt": "Summarize the main points in a clear, organized bullet list",
                "description": "Creates a structured summary with bullet points"
            },
            {
                "name": "Extract Key Information",
                "prompt": "Extract the key information, important dates, names, and decisions mentioned",
                "description": "Focuses on extracting specific facts and details"
            },
            {
                "name": "Meeting Minutes",
                "prompt": "Format this as professional meeting minutes with agenda items, decisions, and action items",
                "description": "Converts to formal meeting minutes format"
            },
            {
                "name": "Q&A Format",
                "prompt": "Organize this content into a question and answer format, highlighting the main topics discussed",
                "description": "Restructures content as Q&A"
            },
            {
                "name": "Executive Summary",
                "prompt": "Create a concise executive summary highlighting the most important points and conclusions",
                "description": "Creates a high-level executive summary"
            },
            {
                "name": "Action Items",
                "prompt": "Extract all action items, tasks, and next steps mentioned in the transcription",
                "description": "Focuses on actionable items and follow-ups"
            },
            {
                "name": "Timeline",
                "prompt": "Create a chronological timeline of events or topics discussed",
                "description": "Organizes content chronologically"
            }
        ] 