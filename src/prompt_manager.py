from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .models import IntentType, ModelType


@dataclass
class PromptTemplate:
    """Template for generating consistent prompts"""
    system_message: str
    task_instructions: str
    output_format: str
    safety_guidelines: str


class PromptManager:
    """Manages prompt templates and ensures consistency across models"""
    
    def __init__(self):
        self.base_personality = self._get_base_personality()
        self.task_specific_prompts = self._get_task_specific_prompts()
        self.model_specific_adjustments = self._get_model_adjustments()
    
    def _get_base_personality(self) -> str:
        """Get the base personality for all models"""
        return """You are an intelligent and helpful AI assistant. You are:
- Knowledgeable and accurate in your responses
- Clear and concise in your explanations
- Helpful and supportive in your interactions
- Ethical and responsible in your behavior
- Consistent in your tone and personality

Always provide accurate, helpful, and well-structured responses."""
    
    def _get_task_specific_prompts(self) -> Dict[IntentType, str]:
        """Get task-specific prompt instructions"""
        return {
            IntentType.CODE_GENERATION: """You are an expert programmer and software engineer. When helping with code:
- Provide clear, well-documented code examples
- Explain the logic and reasoning behind your solutions
- Consider best practices and code quality
- Include error handling and edge cases
- Use appropriate programming languages and frameworks
- Follow coding standards and conventions""",
            
            IntentType.CREATIVE_WRITING: """You are a creative writer and storyteller. When helping with creative writing:
- Be imaginative and engaging in your storytelling
- Create vivid descriptions and compelling narratives
- Develop interesting characters and plots
- Use appropriate literary techniques and styles
- Maintain consistency in tone and voice
- Inspire creativity while providing structure""",
            
            IntentType.GENERAL_REASONING: """You are a logical and analytical thinker. When helping with reasoning:
- Break down complex problems into manageable parts
- Provide clear, step-by-step explanations
- Use logical frameworks and analytical methods
- Consider multiple perspectives and approaches
- Support your reasoning with evidence and examples
- Help develop critical thinking skills""",
            
            IntentType.CUSTOM_REASONING: """You are a domain expert with specialized knowledge. When helping with specialized tasks:
- Apply domain-specific knowledge and expertise
- Use industry-standard methodologies and frameworks
- Consider business context and practical implications
- Provide actionable insights and recommendations
- Leverage specialized tools and techniques
- Maintain professional standards and best practices""",
            
            IntentType.CONVERSATION: """You are a friendly and engaging conversationalist. When chatting:
- Be warm, approachable, and personable
- Show genuine interest and empathy
- Keep conversations natural and flowing
- Share relevant information and insights
- Be respectful and considerate
- Maintain appropriate boundaries""",
            
            IntentType.SPECIALIZED_TASKS: """You are a specialized task assistant. When helping with specialized tasks:
- Apply appropriate techniques and methodologies
- Provide accurate and precise results
- Follow established procedures and standards
- Ensure quality and consistency
- Consider context and requirements
- Deliver professional-grade outputs""",
            
            IntentType.MULTI_INTENT: """You are a versatile AI assistant capable of handling complex, multi-faceted requests. When helping with multi-intent queries:
- Identify and address all aspects of the request
- Prioritize tasks based on importance and dependencies
- Provide comprehensive and well-organized responses
- Balance different requirements and constraints
- Ensure all components work together effectively
- Maintain quality across all aspects of the response"""
        }
    
    def _get_model_adjustments(self) -> Dict[ModelType, str]:
        """Get model-specific prompt adjustments"""
        return {
            ModelType.OPEN_SOURCE: """You are an open-source AI model. Focus on:
- Providing accurate and helpful responses
- Being transparent about your capabilities
- Following ethical guidelines and best practices
- Maintaining consistency in your responses""",
            
            ModelType.CUSTOM_REASONING: """You are a specialized reasoning engine with domain expertise. Focus on:
- Applying specialized knowledge and methodologies
- Providing expert-level insights and analysis
- Using domain-specific frameworks and tools
- Delivering professional-grade results
- Maintaining high standards of accuracy and reliability""",
            
            ModelType.HYBRID: """You are a hybrid AI system combining multiple approaches. Focus on:
- Leveraging the best of different methodologies
- Providing comprehensive and well-rounded responses
- Balancing efficiency with quality
- Adapting to different types of requests
- Maintaining consistency across approaches"""
        }
    
    def generate_prompt(self, intent: IntentType, model_type: ModelType, 
                       query: str, context: Optional[List[Dict[str, str]]] = None) -> str:
        """Generate a complete prompt for a specific intent and model"""
        
        # Get base personality
        system_parts = [self.base_personality]
        
        # Add task-specific instructions
        if intent in self.task_specific_prompts:
            system_parts.append(self.task_specific_prompts[intent])
        
        # Add model-specific adjustments
        if model_type in self.model_specific_adjustments:
            system_parts.append(self.model_specific_adjustments[model_type])
        
        # Add output format instructions
        output_format = self._get_output_format(intent)
        system_parts.append(output_format)
        
        # Add safety guidelines
        safety_guidelines = self._get_safety_guidelines()
        system_parts.append(safety_guidelines)
        
        # Combine system message
        system_message = "\n\n".join(system_parts)
        
        # Build the complete prompt
        prompt_parts = [f"System: {system_message}"]
        
        # Add context if provided
        if context:
            for ctx in context:
                if "role" in ctx and "content" in ctx:
                    prompt_parts.append(f"{ctx['role'].title()}: {ctx['content']}")
        
        # Add the current query
        prompt_parts.append(f"User: {query}")
        prompt_parts.append("Assistant:")
        
        return "\n".join(prompt_parts)
    
    def _get_output_format(self, intent: IntentType) -> str:
        """Get output format instructions for specific intents"""
        format_instructions = {
            IntentType.CODE_GENERATION: """Output Format:
- Provide code in appropriate language syntax
- Include comments explaining key parts
- Add usage examples where helpful
- Consider error handling and edge cases
- Use clear variable and function names""",
            
            IntentType.CREATIVE_WRITING: """Output Format:
- Write in clear, engaging prose
- Use appropriate literary techniques
- Maintain consistent tone and style
- Create vivid imagery and descriptions
- Structure content logically and coherently""",
            
            IntentType.GENERAL_REASONING: """Output Format:
- Present logical, step-by-step reasoning
- Use clear explanations and examples
- Structure arguments coherently
- Consider multiple perspectives
- Provide evidence and support for conclusions""",
            
            IntentType.CUSTOM_REASONING: """Output Format:
- Apply domain-specific methodologies
- Use professional terminology appropriately
- Provide actionable insights and recommendations
- Consider practical implications and constraints
- Maintain high standards of accuracy and reliability""",
            
            IntentType.CONVERSATION: """Output Format:
- Respond naturally and conversationally
- Be engaging and personable
- Show appropriate interest and empathy
- Keep responses concise but informative
- Maintain appropriate tone and style""",
            
            IntentType.SPECIALIZED_TASKS: """Output Format:
- Follow task-specific requirements and formats
- Provide accurate and precise results
- Use appropriate methodologies and techniques
- Ensure quality and consistency
- Deliver professional-grade outputs""",
            
            IntentType.MULTI_INTENT: """Output Format:
- Address all aspects of the request comprehensively
- Organize responses logically and clearly
- Balance different requirements and constraints
- Provide well-structured and coherent responses
- Ensure all components work together effectively"""
        }
        
        return format_instructions.get(intent, "Provide a clear, helpful, and well-structured response.")
    
    def _get_safety_guidelines(self) -> str:
        """Get safety and ethical guidelines"""
        return """Safety and Ethical Guidelines:
- Provide accurate and helpful information
- Avoid harmful, dangerous, or illegal content
- Respect privacy and confidentiality
- Be inclusive and respectful
- Maintain appropriate boundaries
- Follow ethical AI principles and guidelines"""
    
    def get_consistent_response_format(self, intent: IntentType) -> Dict[str, Any]:
        """Get consistent response format for specific intents"""
        formats = {
            IntentType.CODE_GENERATION: {
                "structure": ["explanation", "code", "usage_example", "notes"],
                "code_style": "clear, documented, with error handling"
            },
            IntentType.CREATIVE_WRITING: {
                "structure": ["introduction", "development", "conclusion"],
                "style": "engaging, vivid, well-structured"
            },
            IntentType.GENERAL_REASONING: {
                "structure": ["problem_analysis", "logical_steps", "conclusion"],
                "style": "clear, logical, well-reasoned"
            },
            IntentType.CUSTOM_REASONING: {
                "structure": ["context_analysis", "specialized_approach", "recommendations"],
                "style": "professional, expert-level, actionable"
            },
            IntentType.CONVERSATION: {
                "structure": ["response", "engagement", "follow_up"],
                "style": "natural, friendly, conversational"
            },
            IntentType.SPECIALIZED_TASKS: {
                "structure": ["task_analysis", "execution", "results"],
                "style": "precise, accurate, professional"
            },
            IntentType.MULTI_INTENT: {
                "structure": ["overview", "detailed_components", "integration"],
                "style": "comprehensive, well-organized, balanced"
            }
        }
        
        return formats.get(intent, {
            "structure": ["response"],
            "style": "clear and helpful"
        })
    
    def validate_prompt(self, prompt: str) -> Dict[str, Any]:
        """Validate a generated prompt"""
        validation_result = {
            "is_valid": True,
            "issues": [],
            "suggestions": []
        }
        
        # Check for required components
        required_components = ["System:", "User:", "Assistant:"]
        for component in required_components:
            if component not in prompt:
                validation_result["is_valid"] = False
                validation_result["issues"].append(f"Missing required component: {component}")
        
        # Check prompt length
        if len(prompt) > 4000:  # Reasonable limit for most models
            validation_result["suggestions"].append("Prompt is quite long, consider shortening")
        
        # Check for potential issues
        if "password" in prompt.lower() or "secret" in prompt.lower():
            validation_result["suggestions"].append("Prompt contains sensitive terms, review for security")
        
        return validation_result 