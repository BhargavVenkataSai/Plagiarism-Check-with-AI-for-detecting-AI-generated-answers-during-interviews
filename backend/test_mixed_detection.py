"""
Test script for sentence-level detection on mixed human/AI content.
This demonstrates the ability to detect which sentences are AI vs human.
"""

import asyncio
from detector.classifier import get_detector

async def test_mixed_content():
    # Load the detector
    print("Loading detector...")
    detector = await get_detector()
    
    # Test case 1: Mixed human and AI content
    mixed_text = """I really enjoyed working on that project last summer. 
As an AI language model, I must inform you that implementing best practices is crucial. 
My team and I spent countless hours debugging the issue together. 
The solution involves leveraging cutting-edge paradigms to ensure scalability and maintainability. 
Honestly, I learned a lot from that experience and would do it again."""
    
    print("\n" + "="*80)
    print("TEST 1: Mixed Human + AI Content")
    print("="*80)
    print(f"Input text:\n{mixed_text}\n")
    
    result = await detector.predict_sentences(mixed_text)
    
    print(f"Overall Score: {result['overall_score']:.3f}")
    print(f"Summary: {result['summary']}")
    print(f"Total Sentences: {result['total_sentences']}")
    print(f"  - Human: {result['human_sentences']}")
    print(f"  - AI: {result['ai_sentences']}")
    print(f"  - Uncertain: {result['uncertain_sentences']}")
    print("\nSentence-by-Sentence Analysis:")
    print("-" * 80)
    
    for i, sent_result in enumerate(result['sentences'], 1):
        label_color = {
            'Human': '🟢',
            'AI': '🔴',
            'Uncertain': '🟡'
        }
        icon = label_color.get(sent_result['label'], '⚪')
        print(f"{i}. [{sent_result['label']}] {icon} Score: {sent_result['score']:.3f}")
        print(f"   \"{sent_result['text'][:100]}{'...' if len(sent_result['text']) > 100 else ''}\"")
        print()
    
    # Test case 2: Fully human content
    print("\n" + "="*80)
    print("TEST 2: Fully Human Content")
    print("="*80)
    
    human_text = """I worked at a startup for two years. 
We had some crazy deadlines. 
One time our server crashed at 3am and I had to fix it. 
That experience taught me a lot about staying calm under pressure."""
    
    print(f"Input text:\n{human_text}\n")
    result2 = await detector.predict_sentences(human_text)
    print(f"Summary: {result2['summary']}")
    print(f"Overall Score: {result2['overall_score']:.3f} (lower = more human)")
    
    # Test case 3: Fully AI content
    print("\n" + "="*80)
    print("TEST 3: Fully AI-Generated Content")
    print("="*80)
    
    ai_text = """As an AI language model, I can provide a comprehensive analysis. 
The solution involves implementing a holistic framework that leverages cutting-edge technologies. 
This approach ensures scalability, maintainability, and optimal performance across all scenarios. 
It is important to consider best practices and industry standards when developing such systems."""
    
    print(f"Input text:\n{ai_text}\n")
    result3 = await detector.predict_sentences(ai_text)
    print(f"Summary: {result3['summary']}")
    print(f"Overall Score: {result3['overall_score']:.3f} (higher = more AI)")

if __name__ == "__main__":
    asyncio.run(test_mixed_content())
