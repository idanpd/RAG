"""
Token Budgeting and Prompt Assembly for Conversational RAG
Implements industry-standard token management and prompt construction.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

from conversation_memory import ConversationMessage
from chunker import SemanticChunker, TokenCounter
from utils import ConfigManager

logger = logging.getLogger(__name__)


@dataclass
class TokenBudget:
    """Token budget allocation for different prompt sections."""
    system_prompt: int = 200
    sliding_window: int = 800
    memory_context: int = 600
    doc_context: int = 1000
    user_input: int = 300
    output_reserve: int = 1200
    
    @property
    def total_budget(self) -> int:
        return (self.system_prompt + self.sliding_window + self.memory_context + 
                self.doc_context + self.user_input + self.output_reserve)


@dataclass
class PromptSection:
    """A section of the assembled prompt with metadata."""
    name: str
    content: str
    token_count: int
    priority: int  # Higher = more important
    source_ids: List[str]  # IDs of source items


class PromptAssembler:
    """Assembles prompts with token budgeting and adaptive trimming."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.token_counter = TokenCounter()
        
        # Token budgets (configurable)
        self.budget = TokenBudget(
            system_prompt=config.get('BUDGET_SYSTEM', 200),
            sliding_window=config.get('BUDGET_SLIDING_WINDOW', 800),
            memory_context=config.get('BUDGET_MEMORY', 600),
            doc_context=config.get('BUDGET_DOCS', 1000),
            user_input=config.get('BUDGET_USER_INPUT', 300),
            output_reserve=config.get('BUDGET_OUTPUT_RESERVE', 1200)
        )
        
        # Model context window
        self.max_context = config.get('LLM_CTX', 2048)
        
        # Prompt templates
        self.templates = {
            'system': self._get_system_template(),
            'conversation_turn': "User: {user_msg}\nAssistant: {assistant_msg}\n",
            'memory_item': "Previous context: {content}\n",
            'doc_item': "Document ({source}): {content}\n",
            'final_query': "User: {query}\nAssistant: "
        }
        
        logger.info(f"PromptAssembler initialized with {self.budget.total_budget} token budget")
    
    def _get_system_template(self) -> str:
        """Get the system prompt template."""
        return """You are a helpful AI assistant with access to documents and conversation history. 

Instructions:
- Answer questions using the provided context from documents and conversation history
- Cite sources using [doc:filename] for documents and [memory:turn] for conversation context
- If information is not in the provided context, say so clearly
- Be concise but comprehensive
- Maintain conversation continuity using the chat history

"""
    
    def assemble_prompt(self,
                       query: str,
                       sliding_window: List[ConversationMessage],
                       memory_items: List[Tuple[ConversationMessage, float]],
                       doc_results: List[Dict[str, Any]],
                       conversation_id: str) -> Tuple[str, Dict[str, Any]]:
        """Assemble a complete prompt with token budgeting."""
        
        # Create prompt sections
        sections = []
        
        # 1. System prompt
        system_content = self.templates['system']
        system_tokens = self.token_counter.count_tokens(system_content)
        sections.append(PromptSection(
            name='system',
            content=system_content,
            token_count=system_tokens,
            priority=100,  # Highest priority
            source_ids=[]
        ))
        
        # 2. Sliding window (recent conversation)
        sliding_window_content = self._build_sliding_window(sliding_window)
        sliding_window_tokens = self.token_counter.count_tokens(sliding_window_content)
        sections.append(PromptSection(
            name='sliding_window',
            content=sliding_window_content,
            token_count=sliding_window_tokens,
            priority=90,
            source_ids=[msg.id for msg in sliding_window]
        ))
        
        # 3. Memory context
        memory_content, memory_source_ids = self._build_memory_context(memory_items)
        memory_tokens = self.token_counter.count_tokens(memory_content)
        if memory_content:
            sections.append(PromptSection(
                name='memory',
                content=memory_content,
                token_count=memory_tokens,
                priority=70,
                source_ids=memory_source_ids
            ))
        
        # 4. Document context
        doc_content, doc_source_ids = self._build_doc_context(doc_results)
        doc_tokens = self.token_counter.count_tokens(doc_content)
        if doc_content:
            sections.append(PromptSection(
                name='documents',
                content=doc_content,
                token_count=doc_tokens,
                priority=80,
                source_ids=doc_source_ids
            ))
        
        # 5. Current query
        query_content = self.templates['final_query'].format(query=query)
        query_tokens = self.token_counter.count_tokens(query_content)
        sections.append(PromptSection(
            name='query',
            content=query_content,
            token_count=query_tokens,
            priority=100,  # Highest priority
            source_ids=[]
        ))
        
        # Apply token budgeting
        final_sections, metadata = self._apply_token_budgeting(sections, query)
        
        # Assemble final prompt
        prompt_parts = [section.content for section in final_sections]
        final_prompt = ''.join(prompt_parts)
        
        # Add metadata
        metadata.update({
            'conversation_id': conversation_id,
            'total_tokens': sum(section.token_count for section in final_sections),
            'budget_used': {
                section.name: section.token_count for section in final_sections
            }
        })
        
        return final_prompt, metadata
    
    def _build_sliding_window(self, messages: List[ConversationMessage]) -> str:
        """Build sliding window context from recent messages."""
        if not messages:
            return ""
        
        window_parts = []
        for msg in messages:
            if msg.type == 'user_msg':
                window_parts.append(f"User: {msg.content}")
            elif msg.type == 'assistant_msg':
                # Include confidence and citations info
                content = msg.content
                if msg.citations:
                    citations_str = ", ".join(msg.citations)
                    content += f" [Sources: {citations_str}]"
                window_parts.append(f"Assistant: {content}")
        
        return "\n".join(window_parts) + "\n\n"
    
    def _build_memory_context(self, memory_items: List[Tuple[ConversationMessage, float]]) -> Tuple[str, List[str]]:
        """Build memory context from retrieved conversation history."""
        if not memory_items:
            return "", []
        
        memory_parts = []
        source_ids = []
        
        for msg, score in memory_items:
            source_ids.append(msg.id)
            
            if msg.type == 'summary':
                memory_parts.append(f"Summary: {msg.content}")
            elif msg.type == 'user_msg':
                memory_parts.append(f"Previous user question: {msg.content}")
            elif msg.type == 'assistant_msg':
                # Only include high-confidence assistant messages
                if msg.confidence >= 0.7 or msg.citations:
                    content = msg.content
                    if msg.citations:
                        citations_str = ", ".join(msg.citations)
                        content += f" [Sources: {citations_str}]"
                    memory_parts.append(f"Previous assistant response: {content}")
        
        if memory_parts:
            return "Relevant conversation history:\n" + "\n".join(memory_parts) + "\n\n", source_ids
        return "", []
    
    def _build_doc_context(self, doc_results: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
        """Build document context from search results."""
        if not doc_results:
            return "", []
        
        doc_parts = []
        source_ids = []
        
        for result in doc_results:
            source_ids.append(str(result.get('chunk_id', '')))
            
            path = result.get('path', 'Unknown')
            filename = path.split('/')[-1] if path else 'Unknown'
            content = result.get('text', '')
            
            doc_parts.append(f"[doc:{filename}] {content}")
        
        if doc_parts:
            return "Relevant documents:\n" + "\n".join(doc_parts) + "\n\n", source_ids
        return "", []
    
    def _apply_token_budgeting(self, sections: List[PromptSection], query: str) -> Tuple[List[PromptSection], Dict[str, Any]]:
        """Apply token budgeting and adaptive trimming."""
        
        # Calculate total tokens
        total_tokens = sum(section.token_count for section in sections)
        available_budget = self.max_context - self.budget.output_reserve
        
        metadata = {
            'original_tokens': total_tokens,
            'available_budget': available_budget,
            'trimming_applied': False,
            'sections_modified': []
        }
        
        # If within budget, return as-is
        if total_tokens <= available_budget:
            logger.info(f"Prompt within budget: {total_tokens}/{available_budget} tokens")
            return sections, metadata
        
        logger.info(f"Prompt over budget: {total_tokens}/{available_budget} tokens, applying trimming")
        metadata['trimming_applied'] = True
        
        # Sort sections by priority (lower priority first for trimming)
        trimmable_sections = [s for s in sections if s.name in ['documents', 'memory', 'sliding_window']]
        trimmable_sections.sort(key=lambda x: x.priority)
        
        # Trim sections until within budget
        for section in trimmable_sections:
            if total_tokens <= available_budget:
                break
            
            # Calculate how much to trim
            excess_tokens = total_tokens - available_budget
            
            if section.name == 'documents':
                # Trim documents first (drop lowest relevance)
                trimmed_content, trimmed_tokens = self._trim_documents(section, excess_tokens)
                tokens_saved = section.token_count - trimmed_tokens
                
            elif section.name == 'memory':
                # Trim memory items
                trimmed_content, trimmed_tokens = self._trim_memory(section, excess_tokens)
                tokens_saved = section.token_count - trimmed_tokens
                
            elif section.name == 'sliding_window':
                # Summarize older parts of sliding window
                trimmed_content, trimmed_tokens = self._trim_sliding_window(section, excess_tokens)
                tokens_saved = section.token_count - trimmed_tokens
            
            else:
                continue
            
            # Update section
            section.content = trimmed_content
            section.token_count = trimmed_tokens
            total_tokens -= tokens_saved
            
            metadata['sections_modified'].append({
                'section': section.name,
                'tokens_saved': tokens_saved,
                'new_token_count': trimmed_tokens
            })
            
            logger.info(f"Trimmed {section.name}: saved {tokens_saved} tokens")
        
        # Final check
        final_tokens = sum(section.token_count for section in sections)
        metadata['final_tokens'] = final_tokens
        
        if final_tokens > available_budget:
            logger.warning(f"Still over budget after trimming: {final_tokens}/{available_budget}")
        
        return sections, metadata
    
    def _trim_documents(self, section: PromptSection, target_reduction: int) -> Tuple[str, int]:
        """Trim document section by removing lowest priority items."""
        lines = section.content.split('\n')
        doc_lines = [line for line in lines if line.strip() and line.startswith('[doc:')]
        
        if not doc_lines:
            return section.content, section.token_count
        
        # Remove documents from the end (assuming they're ordered by relevance)
        tokens_saved = 0
        kept_lines = []
        
        for line in doc_lines:
            line_tokens = self.token_counter.count_tokens(line)
            if tokens_saved < target_reduction:
                tokens_saved += line_tokens
            else:
                kept_lines.append(line)
        
        # Rebuild content
        if kept_lines:
            new_content = "Relevant documents:\n" + "\n".join(kept_lines) + "\n\n"
        else:
            new_content = ""
        
        new_tokens = self.token_counter.count_tokens(new_content)
        return new_content, new_tokens
    
    def _trim_memory(self, section: PromptSection, target_reduction: int) -> Tuple[str, int]:
        """Trim memory section by removing lowest relevance items."""
        lines = section.content.split('\n')
        memory_lines = [line for line in lines if line.strip() and 
                       (line.startswith('Previous') or line.startswith('Summary'))]
        
        if not memory_lines:
            return section.content, section.token_count
        
        # Remove from the end (lowest relevance)
        tokens_saved = 0
        kept_lines = []
        
        for line in reversed(memory_lines):
            line_tokens = self.token_counter.count_tokens(line)
            if tokens_saved < target_reduction:
                tokens_saved += line_tokens
            else:
                kept_lines.insert(0, line)  # Insert at beginning to maintain order
        
        # Rebuild content
        if kept_lines:
            new_content = "Relevant conversation history:\n" + "\n".join(kept_lines) + "\n\n"
        else:
            new_content = ""
        
        new_tokens = self.token_counter.count_tokens(new_content)
        return new_content, new_tokens
    
    def _trim_sliding_window(self, section: PromptSection, target_reduction: int) -> Tuple[str, int]:
        """Trim sliding window by summarizing older exchanges."""
        lines = section.content.split('\n')
        exchanges = []
        current_exchange = []
        
        for line in lines:
            if line.strip():
                if line.startswith('User:') and current_exchange:
                    exchanges.append(current_exchange)
                    current_exchange = [line]
                else:
                    current_exchange.append(line)
        
        if current_exchange:
            exchanges.append(current_exchange)
        
        if len(exchanges) <= 1:
            return section.content, section.token_count
        
        # Keep the most recent exchange, summarize older ones
        recent_exchange = exchanges[-1]
        older_exchanges = exchanges[:-1]
        
        # Create summary of older exchanges
        summary_text = f"Earlier in conversation: User asked about various topics and received responses."
        
        # Rebuild content
        summary_lines = [summary_text] if older_exchanges else []
        recent_lines = recent_exchange
        
        new_lines = summary_lines + recent_lines
        new_content = '\n'.join(new_lines) + '\n\n'
        new_tokens = self.token_counter.count_tokens(new_content)
        
        return new_content, new_tokens
    
    def estimate_response_tokens(self, query: str) -> int:
        """Estimate how many tokens the response might need."""
        query_tokens = self.token_counter.count_tokens(query)
        
        # Heuristic: response is usually 1-3x query length, with minimum reserve
        estimated = max(query_tokens * 2, 300)
        
        return min(estimated, self.budget.output_reserve)
    
    def validate_prompt(self, prompt: str) -> Dict[str, Any]:
        """Validate the assembled prompt."""
        prompt_tokens = self.token_counter.count_tokens(prompt)
        
        validation = {
            'valid': True,
            'token_count': prompt_tokens,
            'within_budget': prompt_tokens <= (self.max_context - self.budget.output_reserve),
            'estimated_total': prompt_tokens + self.budget.output_reserve,
            'warnings': []
        }
        
        if not validation['within_budget']:
            validation['valid'] = False
            validation['warnings'].append(f"Prompt too long: {prompt_tokens} tokens")
        
        if prompt_tokens < 100:
            validation['warnings'].append("Prompt seems very short")
        
        return validation