
from typing import Optional

def normalize_prompt_text(text: str) -> str:
    """
    Normalize prompt text.
    Restored as fallback for missing file.
    """
    if not text:
        return ""
    return text.strip()

def render_thinking(
    task_block: str,
    safety_block: str,
    policy_block: str,
    analysis_block: Optional[str] = None,
    use_quadruple: bool = True,
    style: str = "xml"
) -> str:
    """
    Render VSSI Quadruple Reasoning thinking block.
    
    Args:
        task_block: Vector State (Observation/Fact)
        safety_block: Negative Spinor (Abduction/Safety)
        policy_block: Quadrality Integration (Synthesis/Policy)
        analysis_block: Positive Spinor (Deduction/Logic) - Optional
        use_quadruple: Whether to use 4-way tags or collapsed tags
        style: Rendering style (default: 'xml')
        
    Returns:
        Formatted string enclosed in <think> tags or specific VSSI tags.
    """
    if style == "xml":
        # Full Quadruple Reasoning XML format
        parts = []
        parts.append("<think>")
        
        # 1. Task/Observation
        parts.append(f"<think-task>\n{task_block}\n</think-task>")
        
        # 2. Analysis/Logic (Positive Spinor)
        if analysis_block:
            parts.append(f"<think-analysis>\n{analysis_block}\n</think-analysis>")
            
        # 3. Safety/Edges (Negative Spinor)
        parts.append(f"<think-safety>\n{safety_block}\n</think-safety>")
        
        # 4. Policy/Integration
        parts.append(f"<think-policy>\n{policy_block}\n</think-policy>")
        
        parts.append("</think>")
        return "\n".join(parts)
        
    else:
        # Fallback simplistic format
        content = f"Task: {task_block}\n"
        if analysis_block:
            content += f"Analysis: {analysis_block}\n"
        content += f"Safety: {safety_block}\n"
        content += f"Policy: {policy_block}"
        return f"<think>\n{content}\n</think>"
