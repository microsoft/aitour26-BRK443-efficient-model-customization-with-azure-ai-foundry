"""Format document tags in user messages."""

import re


def extract_documents(message: str) -> list:
    """
    Extract all documents from a message containing <DOCUMENT> tags.
    
    Args:
        message: Message containing <DOCUMENT> tags
        
    Returns:
        List of document contents (stripped of whitespace)
    """
    # Pattern to match document tags and their content
    pattern = r'<DOCUMENT>(.*?)</DOCUMENT>'
    
    # Find all documents
    documents = re.findall(pattern, message, re.DOTALL)
    
    # Clean up each document (remove extra whitespace)
    cleaned_documents = [doc.strip() for doc in documents]
    
    return cleaned_documents


def reformat_user_documents(message: str) -> str:
    """
    Reformat user message with <DOCUMENT> tags into numbered sources format.
    
    Args:
        message: User message containing <DOCUMENT> tags
        
    Returns:
        Reformatted message with numbered sources
    """
    # Pattern to match document tags and their content
    pattern = r'<DOCUMENT>(.*?)</DOCUMENT>'
    
    # Find all documents
    documents = re.findall(pattern, message, re.DOTALL)
    
    if not documents:
        # No documents found, return as is
        return message
    
    # Extract the question (everything after the last </DOCUMENT>)
    parts = re.split(r'</DOCUMENT>', message)
    question = parts[-1].strip() if len(parts) > 1 else ''
    
    # Build formatted output
    formatted_lines = ["Sources:"]
    
    for i, doc in enumerate(documents, 1):
        # Clean up the document content (remove extra whitespace)
        cleaned_doc = doc.strip()
        formatted_lines.append(f"  - [{i}] {cleaned_doc}")
    
    # Add the question at the end
    if question:
        formatted_lines.append("")  # Empty line before question
        formatted_lines.append(question)
    
    return '\n'.join(formatted_lines)


def reformat_quotes_with_citations(message: str, documents: list) -> str:
    """
    Reformat quotes with ##begin_quote## and ##end_quote## markers to use backticks and citations.
    
    Args:
        message: Message containing quote markers
        documents: List of document strings to match quotes against
        
    Returns:
        Reformatted message with quotes in backticks and citation numbers
    """
    # Pattern to match quotes
    pattern = r'##begin_quote##(.*?)##end_quote##'
    
    def normalize_text(text):
        """Normalize text by removing extra whitespace and newlines for comparison."""
        return ' '.join(text.split())
    
    def replace_quote(match):
        quote_text = match.group(1).strip()
        normalized_quote = normalize_text(quote_text)
        
        # Find which document contains this quote
        citation_num = None
        for i, doc in enumerate(documents, 1):
            normalized_doc = normalize_text(doc)
            if normalized_quote in normalized_doc:
                citation_num = i
                break
        
        # Format with backticks and citation
        if citation_num:
            return f'`{quote_text}` [{citation_num}]'
        else:
            # No matching document found, just format without citation
            return f'`{quote_text}`'
    
    # Replace all quotes
    reformatted = re.sub(pattern, replace_quote, message, flags=re.DOTALL)
    
    return reformatted


# Example usage
if __name__ == "__main__":
    test_message = """<DOCUMENT>I have used a few different kinds of sprayers at all different price points, and all have been easy to use and have given 
an impeccable finish. Step 6: Sand and Wipe Down Again
KILZ 3® PREMIUM Primer is ready for paint in just one hour. After the primer has dried, lightly sand the surfaces once more to ensure a smooth finish.</DOCUMENT>
<DOCUMENT>High-VOC paints release harmful chemicals that affect people, plants, animals, 
and aquatic life during application and curing. Prioritize low-VOC or zero-VOC formulations for outdoor furniture projects. These environmentally responsible options protect your family's health while 
minimizing ecological impact.</DOCUMENT>
<DOCUMENT>Ensure adequate ventilation even in outdoor spaces, 
particularly when using oil-based products. Step 2: Thorough Furniture Cleaning 
Deep cleaning forms the foundation of professional results. For weathered outdoor furniture, use soap, water, and scrub brushes to remove accumulated 
dirt, grime, and surface contaminants. This process eliminates barriers that prevent proper paint adhesion.</DOCUMENT>
<DOCUMENT>Remember that outdoor-specific paints should remain outdoors.</DOCUMENT>
What type of paint should be prioritized for outdoor furniture projects?"""
    
    formatted = reformat_user_documents(test_message)
    print(formatted)