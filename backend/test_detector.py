"""
Test script for AI detection system.
Run with: python test_detector.py

Make sure backend is running: uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""
import requests
import json

BACKEND_URL = "http://localhost:8000"

# Test cases: (text, expected_label, description)
test_cases = [
    # Clear HUMAN samples
    (
        "Honestly I'm not sure about that one. Let me think about it.",
        "Human",
        "Informal, uncertain"
    ),
    (
        "I debugged it by adding some logs and found the bug after lunch.",
        "Human",
        "Casual, personal experience"
    ),
    (
        "We had a tight deadline so I just hacked something together. Not proud of it.",
        "Human",
        "Informal, self-deprecating"
    ),
    (
        "My manager wasn't happy but we shipped it anyway.",
        "Human",
        "Short, conversational"
    ),
    (
        "I learned React by building side projects. Took me a few months.",
        "Human",
        "Personal learning story"
    ),
    
    # Clear AI samples
    (
        "As an AI language model, I don't have personal experiences or emotions.",
        "AI",
        "Explicit AI statement"
    ),
    (
        "Based on the comprehensive analysis of requirements, I recommend implementing a robust, scalable architecture following industry best practices and established design patterns.",
        "AI",
        "Formal, comprehensive"
    ),
    (
        "The optimal solution involves implementing a holistic framework that addresses all aspects of the problem while ensuring maintainability and scalability.",
        "AI",
        "Buzzwords, formal structure"
    ),
    (
        "Here is a step-by-step approach to solving this complex problem effectively, considering all stakeholders and requirements.",
        "AI",
        "Structured, comprehensive"
    ),
    (
        "I would recommend a methodology that incorporates best practices, robust error handling, and comprehensive testing strategies.",
        "AI",
        "Formal recommendations"
    ),
    
    # EDGE CASES (harder to classify)
    (
        "I implemented a microservices architecture to improve scalability.",
        "Edge",
        "Technical but could be either"
    ),
    (
        "Yeah, I'd probably start by breaking that down into smaller chunks first.",
        "Edge",
        "Casual but could be AI mimicking"
    ),
    (
        "The best approach would be to analyze the requirements carefully first.",
        "Edge",
        "Generic advice"
    ),
    (
        "I think testing is important for any production system.",
        "Edge",
        "Simple statement"
    ),
]

def test_detect(text: str) -> dict:
    """Call /detect endpoint."""
    try:
        resp = requests.post(f"{BACKEND_URL}/detect", json={"text": text}, timeout=30)
        return resp.json()
    except Exception as e:
        return {"error": str(e)}

def test_detect_debug(text: str) -> dict:
    """Call /detect/debug endpoint."""
    try:
        resp = requests.post(f"{BACKEND_URL}/detect/debug", json={"text": text}, timeout=30)
        return resp.json()
    except Exception as e:
        return {"error": str(e)}

def score_to_label(score: float) -> str:
    if score < 0.4:
        return "Human"
    elif score > 0.6:
        return "AI"
    else:
        return "Uncertain"

def main():
    print("="*70)
    print("AI DETECTION TEST SUITE")
    print("="*70)
    
    # Check backend health
    try:
        health = requests.get(f"{BACKEND_URL}/health", timeout=5).json()
        print(f"Backend status: {health['status']}")
    except:
        print("ERROR: Backend not running! Start with:")
        print("  uvicorn app:app --host 0.0.0.0 --port 8000 --reload")
        return
    
    print("\n" + "-"*70)
    print("RUNNING TEST CASES")
    print("-"*70)
    
    results = {"correct": 0, "incorrect": 0, "edge": 0}
    
    for i, (text, expected, description) in enumerate(test_cases, 1):
        result = test_detect(text)
        
        if "error" in result:
            print(f"\n#{i} ERROR: {result['error']}")
            continue
        
        score = result["score"]
        predicted = score_to_label(score)
        flag = result["flag"]
        
        # Determine if correct
        if expected == "Edge":
            status = "⚪ EDGE"
            results["edge"] += 1
        elif expected == predicted:
            status = "✅ CORRECT"
            results["correct"] += 1
        elif expected == "Human" and predicted == "Uncertain":
            status = "🟡 BORDERLINE"
            results["correct"] += 1
        elif expected == "AI" and predicted == "Uncertain":
            status = "🟡 BORDERLINE"
            results["correct"] += 1
        else:
            status = "❌ WRONG"
            results["incorrect"] += 1
        
        # Color for score
        if score < 0.4:
            score_color = "🟢"
        elif score > 0.6:
            score_color = "🔴"
        else:
            score_color = "🟡"
        
        print(f"\n#{i} {status}")
        print(f"   Text: \"{text[:60]}{'...' if len(text) > 60 else ''}\"")
        print(f"   Expected: {expected} | Predicted: {predicted} | Score: {score_color} {score:.2f}")
        print(f"   Description: {description}")
    
    # Summary
    total = results["correct"] + results["incorrect"]
    accuracy = results["correct"] / total * 100 if total > 0 else 0
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"  Correct:   {results['correct']}/{total} ({accuracy:.1f}%)")
    print(f"  Incorrect: {results['incorrect']}/{total}")
    print(f"  Edge cases (not scored): {results['edge']}")
    print("="*70)
    
    # Interactive mode
    print("\n" + "-"*70)
    print("INTERACTIVE MODE - Type text to test (or 'quit' to exit)")
    print("-"*70)
    
    while True:
        try:
            text = input("\nEnter text to analyze: ").strip()
            if text.lower() in ['quit', 'exit', 'q']:
                break
            if not text:
                continue
            
            print("\nAnalyzing...")
            result = test_detect_debug(text)
            
            if "error" in result:
                print(f"Error: {result['error']}")
                continue
            
            score = result["score"]
            predicted = score_to_label(score)
            
            print(f"\n{'='*50}")
            print(f"RESULT: {predicted}")
            print(f"{'='*50}")
            print(f"  Score: {score:.3f}")
            print(f"  Flag (AI detected): {result['flag']}")
            print(f"  Confidence: {result.get('confidence', 'N/A')}")
            print(f"  Explanation: {result['explanation']}")
            
            if result.get('model_votes'):
                print(f"\n  Individual model votes:")
                for model, vote in result['model_votes'].items():
                    bar = "█" * int(vote * 20) + "░" * (20 - int(vote * 20))
                    print(f"    {model:4s}: {bar} {vote:.2f}")
            
            if result.get('similar_samples'):
                print(f"\n  Most similar training samples:")
                for j, sample in enumerate(result['similar_samples'][:3], 1):
                    print(f"    {j}. [{sample['label']}] (sim: {sample['similarity']:.2f})")
                    print(f"       \"{sample['text'][:80]}...\"")
            
        except KeyboardInterrupt:
            break
    
    print("\nGoodbye!")

if __name__ == "__main__":
    main()
