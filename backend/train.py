"""
Training script for AI-likeness classifier.
- Uses sentence-transformers/all-MiniLM-L6-v2 for embeddings
- Trains LogisticRegression on human vs AI text samples
- Evaluates with train/test split
- Saves model to detector/model/clf.pkl

Usage (Windows cmd):
    python backend/train.py

Customize:
- Edit `human_texts` and `ai_texts` below or load your own dataset.
"""
import os
import sys
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
try:
    from sentence_transformers import SentenceTransformer
except ModuleNotFoundError:
    print("sentence_transformers not found. Please install backend requirements:")
    print("  pip install -r requirements.txt")
    print("On Windows, also install CPU-only Torch:")
    print("  pip install --index-url https://download.pytorch.org/whl/cpu torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2")
    sys.exit(1)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "detector", "model")
MODEL_PATH = os.path.join(MODEL_DIR, "clf.pkl")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# 100 Human interview answer samples (natural, conversational, imperfect)
human_texts = [
    "I worked on a project where we had to migrate our database to the cloud.",
    "Honestly, I'm not sure about that one. I'd need to look it up.",
    "My approach was to break the problem into smaller pieces and tackle each one.",
    "I collaborated with the frontend team to fix a critical bug before release.",
    "We had a tight deadline, so I prioritized the most important features first.",
    "I learned Python on my own through online tutorials and practice projects.",
    "The hardest part was convincing stakeholders to change their requirements.",
    "I usually start by understanding the problem before jumping into code.",
    "My manager gave me feedback that I needed to communicate more proactively.",
    "I made a mistake in production once and learned to always double-check configs.",
    "We used Agile methodology, though our sprints were sometimes chaotic.",
    "I prefer working in small teams where communication is easier.",
    "The interview process at my last company was pretty straightforward.",
    "I debugged the issue by adding log statements and tracing the flow.",
    "Sometimes I get stuck and need to take a break before finding the solution.",
    "I've used React for about two years now, mostly for internal tools.",
    "My strength is probably problem-solving, but I can be impatient sometimes.",
    "I asked my colleague for help when I couldn't figure out the algorithm.",
    "We shipped the feature late because of unexpected edge cases.",
    "I think testing is important, but I admit I don't always write enough tests.",
    "The project failed because we underestimated the complexity.",
    "I stayed late to fix a bug that was blocking the release.",
    "I'm still learning about system design; it's a weak area for me.",
    "My team used Git flow, but we often had merge conflicts.",
    "I enjoy mentoring junior developers when I have the time.",
    "I disagreed with my manager once about the technical approach.",
    "We had to pivot the project halfway through due to changing requirements.",
    "I used Stack Overflow a lot when I was learning Django.",
    "The code review process at my company was pretty informal.",
    "I find documentation boring but I know it's necessary.",
    "I've never used Kubernetes in production, only played with it locally.",
    "My biggest achievement was reducing page load time by 40 percent.",
    "I got frustrated when the requirements kept changing mid-sprint.",
    "I prefer backend work but I can do frontend if needed.",
    "We had a postmortem after the outage and identified three root causes.",
    "I think I'm good at debugging but less experienced with architecture.",
    "The team dynamics were difficult because of personality conflicts.",
    "I automated a manual process that saved the team hours each week.",
    "I'm not great at estimating how long tasks will take.",
    "I've done some pair programming but I prefer working independently.",
    "The legacy codebase was a mess and nobody wanted to touch it.",
    "I took ownership of the feature even though it wasn't my area.",
    "I failed a technical interview once because I blanked on recursion.",
    "We used Jenkins for CI/CD but it was always breaking.",
    "I think remote work is fine but I miss in-person collaboration.",
    "I had to learn AWS quickly for a project last year.",
    "My weakness is probably that I overthink solutions sometimes.",
    "I've contributed to open source a few times, nothing major.",
    "The product manager and I had different priorities.",
    "I refactored the authentication module to improve security.",
    "I'm comfortable with SQL but haven't used NoSQL much.",
    "We did daily standups but they often ran too long.",
    "I enjoy solving algorithmic problems, though I'm not the fastest.",
    "I've been coding for about five years, mostly in JavaScript.",
    "The deadline was unrealistic but we managed to ship something.",
    "I try to write clean code but sometimes I cut corners under pressure.",
    "I learned a lot from that project even though it didn't succeed.",
    "My previous company had a toxic culture, which is why I left.",
    "I think communication skills are just as important as technical skills.",
    "I've never managed people directly, but I've led small projects.",
    "The hardest bug I fixed took me three days to track down.",
    "I'm interested in machine learning but haven't had professional experience.",
    "I prefer TypeScript over JavaScript because of the type safety.",
    "We had to support IE11 which was a nightmare.",
    "I've done some DevOps work but I'm not an expert.",
    "I usually ask clarifying questions before starting a task.",
    "The codebase had no tests so I started adding them gradually.",
    "I've worked in startups and large companies, both have pros and cons.",
    "I got better at handling pressure after a few intense projects.",
    "My approach to learning is to build projects and learn by doing.",
    "I've had to give critical feedback to teammates, which was uncomfortable.",
    "The API design was confusing so I suggested improvements.",
    "I'm okay with ambiguity but prefer some level of clarity.",
    "I've done mobile development with React Native for one project.",
    "My manager was supportive and gave me room to grow.",
    "I think code reviews are valuable for catching bugs and sharing knowledge.",
    "I've dealt with imposter syndrome, especially early in my career.",
    "We used Slack for communication but sometimes things got lost.",
    "I prefer to understand the business context before coding.",
    "I've been on-call and handled incidents, though it's stressful.",
    "The project scope kept expanding, which made delivery difficult.",
    "I think soft skills are underrated in engineering.",
    "I've mentored interns and helped them ramp up.",
    "The tech debt accumulated because we never had time to address it.",
    "I've done performance optimization on a slow API endpoint.",
    "My first job out of college was at a small startup.",
    "I think whiteboard interviews don't reflect real work accurately.",
    "I've worked with distributed teams across different time zones.",
    "The product wasn't successful but I learned about user research.",
    "I've used Docker for local development and deployment.",
    "I try to balance speed and quality depending on the situation.",
    "I've had to push back on unrealistic deadlines politely.",
    "The architecture decisions were made before I joined the team.",
    "I enjoy learning new technologies but don't chase every trend.",
    "I've done A/B testing to validate feature hypotheses.",
    "My experience with databases includes PostgreSQL and MySQL.",
    "I think documentation helps onboarding but nobody reads it.",
    "I've attended conferences and found them useful for networking.",
    "The integration with the third-party API was buggy.",
    "I prefer honest feedback even if it's negative.",
]

# 100 AI-generated text samples (formal, comprehensive, structured)
ai_texts = [
    "As an AI language model, I don't have personal experiences or emotions.",
    "The optimal solution involves implementing a comprehensive framework that addresses all aspects of the problem.",
    "Based on the information provided, I can offer a detailed analysis of the situation.",
    "Here is a step-by-step approach to solving this complex problem effectively.",
    "The implementation should follow industry best practices and established design patterns.",
    "I would recommend a holistic approach that considers all stakeholders and requirements.",
    "The architecture should be scalable, maintainable, and adhere to SOLID principles.",
    "Let me provide a comprehensive overview of the key considerations involved.",
    "The solution leverages cutting-edge technologies to deliver optimal performance.",
    "Based on my training data, I can suggest several approaches to address this challenge.",
    "The methodology involves a systematic analysis of requirements and constraints.",
    "Here are the key factors to consider when making this technical decision.",
    "The implementation utilizes a microservices architecture for enhanced scalability.",
    "I can provide detailed guidance on best practices for this particular scenario.",
    "The approach ensures robustness, reliability, and adherence to quality standards.",
    "Let me outline a structured framework for addressing this requirement effectively.",
    "The solution incorporates error handling, logging, and monitoring capabilities.",
    "Based on the context provided, here is my recommended course of action.",
    "The design pattern employed here optimizes for both performance and maintainability.",
    "I would suggest implementing a comprehensive testing strategy including unit and integration tests.",
    "The architecture follows the principle of separation of concerns for modularity.",
    "Here is a detailed breakdown of the components and their interactions.",
    "The implementation adheres to security best practices and compliance requirements.",
    "Let me provide an exhaustive analysis of the trade-offs involved in this decision.",
    "The solution is designed to handle edge cases and exceptional scenarios gracefully.",
    "Based on established methodologies, I recommend the following approach.",
    "The framework provides a robust foundation for future enhancements and extensions.",
    "Here are the critical success factors for implementing this solution effectively.",
    "The architecture employs caching strategies to optimize response times.",
    "I can elaborate on the technical specifications and implementation details.",
    "The design incorporates fault tolerance and disaster recovery mechanisms.",
    "Let me outline the key architectural decisions and their rationale.",
    "The solution utilizes asynchronous processing for improved throughput.",
    "Based on industry standards, here is the recommended implementation approach.",
    "The methodology ensures alignment with organizational goals and objectives.",
    "Here is a comprehensive guide to deploying and maintaining this solution.",
    "The architecture supports horizontal scaling to accommodate growing demands.",
    "I would recommend implementing observability features for operational insights.",
    "The design follows the principle of least privilege for enhanced security.",
    "Let me provide a detailed comparison of the available implementation options.",
    "The solution incorporates automated testing and continuous integration practices.",
    "Based on the requirements, here is an optimized implementation strategy.",
    "The framework enables seamless integration with existing systems and workflows.",
    "Here are the technical prerequisites and dependencies for this implementation.",
    "The architecture employs event-driven patterns for loose coupling between components.",
    "I can provide comprehensive documentation covering all aspects of the solution.",
    "The design prioritizes user experience while maintaining technical excellence.",
    "Let me outline the deployment strategy and rollout considerations.",
    "The solution leverages containerization for consistent deployment across environments.",
    "Based on best practices, I recommend implementing the following safeguards.",
    "The methodology incorporates feedback loops for continuous improvement.",
    "Here is a detailed explanation of the data flow and processing logic.",
    "The architecture supports multi-tenancy for serving diverse user populations.",
    "I would suggest establishing clear metrics and KPIs for measuring success.",
    "The design incorporates rate limiting and throttling for resource protection.",
    "Let me provide an overview of the monitoring and alerting capabilities.",
    "The solution ensures data integrity through transactional guarantees.",
    "Based on the analysis, here are the recommended optimization strategies.",
    "The framework supports extensibility through well-defined plugin interfaces.",
    "Here are the considerations for ensuring high availability and reliability.",
    "The architecture employs the circuit breaker pattern for resilience.",
    "I can elaborate on the security measures implemented in this solution.",
    "The design follows responsive principles for cross-device compatibility.",
    "Let me outline the data migration strategy and rollback procedures.",
    "The solution provides comprehensive audit logging for compliance purposes.",
    "Based on the specifications, here is the recommended technology stack.",
    "The methodology ensures consistency across development and production environments.",
    "Here is a detailed analysis of the performance characteristics and bottlenecks.",
    "The architecture supports blue-green deployments for zero-downtime releases.",
    "I would recommend implementing comprehensive input validation and sanitization.",
    "The design incorporates progressive enhancement for improved accessibility.",
    "Let me provide guidance on capacity planning and resource allocation.",
    "The solution leverages caching at multiple layers for optimal performance.",
    "Based on the requirements analysis, here are the prioritized feature recommendations.",
    "The framework provides abstraction layers for database independence.",
    "Here are the best practices for maintaining code quality and consistency.",
    "The architecture employs the CQRS pattern for read/write optimization.",
    "I can provide detailed documentation on the API endpoints and contracts.",
    "The design supports internationalization and localization requirements.",
    "Let me outline the backup and recovery procedures for data protection.",
    "The solution incorporates machine learning capabilities for intelligent features.",
    "Based on industry benchmarks, here are the performance optimization recommendations.",
    "The methodology ensures traceability from requirements to implementation.",
    "Here is a comprehensive overview of the system's functional capabilities.",
    "The architecture supports real-time data synchronization across components.",
    "I would suggest implementing feature flags for controlled rollouts.",
    "The design follows accessibility guidelines for inclusive user experiences.",
    "Let me provide an analysis of the total cost of ownership considerations.",
    "The solution ensures backward compatibility with existing client implementations.",
    "Based on the evaluation criteria, here is my assessment of the options.",
    "The framework provides hooks for customization without modifying core logic.",
    "Here are the recommended practices for managing technical debt effectively.",
    "The architecture employs message queuing for reliable asynchronous communication.",
    "I can elaborate on the governance model and decision-making processes.",
    "The design incorporates progressive loading for improved perceived performance.",
    "Let me outline the quality assurance processes and acceptance criteria.",
    "The solution provides comprehensive error messages for improved debuggability.",
    "Based on the analysis of requirements, here is the proposed system design.",
    "The methodology incorporates risk assessment and mitigation strategies.",
    "Here is a detailed specification of the integration touchpoints and protocols.",
]

# HARD/EDGE CASES - These blur the line between human and AI
# Human texts that sound formal/polished (could be mistaken for AI)
hard_human_texts = [
    "I implemented a microservices architecture to improve scalability and maintainability.",
    "The solution I developed followed best practices for security and performance optimization.",
    "I conducted a comprehensive analysis of the requirements before designing the system.",
    "My approach involved implementing robust error handling and logging mechanisms.",
    "I ensured the codebase adhered to SOLID principles and clean architecture patterns.",
    "The deployment pipeline I created incorporated automated testing and continuous integration.",
    "I optimized database queries to improve response times by approximately forty percent.",
    "My implementation utilized caching strategies to reduce server load significantly.",
    "I designed the API following RESTful conventions and documented all endpoints thoroughly.",
    "The system I built supports horizontal scaling to handle increased traffic demands.",
    "I implemented comprehensive unit tests achieving ninety percent code coverage.",
    "My solution incorporated monitoring and alerting for production incident detection.",
    "I refactored the legacy codebase to improve maintainability and reduce technical debt.",
    "The architecture I proposed enables seamless integration with third-party services.",
    "I established coding standards and conducted regular code reviews for quality assurance.",
]

# AI texts that sound casual/human-like (could be mistaken for human)
hard_ai_texts = [
    "Yeah, I'd probably start by breaking that down into smaller chunks first.",
    "Hmm, that's tricky. I think the main issue here is the database design.",
    "So basically, you want to cache that data to speed things up.",
    "I'm not totally sure, but I'd guess the problem is with the API rate limiting.",
    "Oh interesting, I've seen similar issues before with authentication flows.",
    "Right, so the quick fix would be adding an index to that table.",
    "Makes sense. I'd probably use Redis for that kind of caching scenario.",
    "Cool, so you'd want to set up a webhook to handle those events.",
    "That's a good point actually. The trade-off is between speed and consistency.",
    "I think the simplest approach would be to use a message queue here.",
    "Fair enough. You could also try lazy loading to improve initial load time.",
    "Got it. So the bottleneck is probably in the network layer somewhere.",
    "Nice, that should work. Just make sure to handle the edge cases properly.",
    "Okay so basically you need to implement retry logic with exponential backoff.",
    "Sure thing. I'd recommend using environment variables for that configuration.",
]

# Mixed/ambiguous texts (genuinely hard to classify)
ambiguous_texts = [
    "The best approach would be to analyze the requirements carefully first.",
    "I think implementing proper error handling is essential for production systems.",
    "We should consider using a caching layer to improve performance.",
    "My recommendation would be to start with a simple solution and iterate.",
    "It's important to write tests to ensure the code works correctly.",
    "The database schema should be designed with scalability in mind.",
    "I suggest using version control and following a branching strategy.",
    "Performance optimization should focus on the most critical bottlenecks.",
    "Clean code practices help maintain the codebase over time.",
    "Security considerations should be addressed early in the development process.",
]
# Labels: 0=human, 1=AI (these are genuinely ambiguous, assigned based on origin)
ambiguous_labels = [1, 0, 1, 0, 0, 1, 0, 1, 0, 1]  # Mixed origins

def main():
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    
    print("Loading embedding model...")
    st_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    # Combine all datasets
    all_human = human_texts + hard_human_texts
    all_ai = ai_texts + hard_ai_texts
    
    # Add ambiguous samples
    ambiguous_human = [ambiguous_texts[i] for i, l in enumerate(ambiguous_labels) if l == 0]
    ambiguous_ai = [ambiguous_texts[i] for i, l in enumerate(ambiguous_labels) if l == 1]
    
    all_human += ambiguous_human
    all_ai += ambiguous_ai
    
    print(f"\nDataset composition:")
    print(f"  Base human samples: {len(human_texts)}")
    print(f"  Hard human samples: {len(hard_human_texts)}")
    print(f"  Base AI samples: {len(ai_texts)}")
    print(f"  Hard AI samples: {len(hard_ai_texts)}")
    print(f"  Ambiguous samples: {len(ambiguous_texts)}")
    print(f"  TOTAL: {len(all_human)} human, {len(all_ai)} AI = {len(all_human) + len(all_ai)} samples")
    
    # Prepare data
    texts = all_human + all_ai
    y = np.array([0] * len(all_human) + [1] * len(all_ai))
    
    print("Encoding texts...")
    X = st_model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    
    # Train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Define multiple models
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, C=1.0),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'MLP': MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42),
    }
    
    print("\n" + "="*60)
    print("CROSS-VALIDATION (5-Fold) - More Realistic Accuracy")
    print("="*60)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_results = {}
    for name, model_class in [
        ('LogisticRegression', lambda: LogisticRegression(max_iter=1000, C=1.0)),
        ('RandomForest', lambda: RandomForestClassifier(n_estimators=100, random_state=42)),
        ('SVM', lambda: SVC(kernel='rbf', probability=True, random_state=42)),
        ('GradientBoosting', lambda: GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ('MLP', lambda: MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)),
    ]:
        scores = cross_val_score(model_class(), X, y, cv=cv, scoring='accuracy')
        cv_results[name] = (scores.mean(), scores.std())
        print(f"  {name:20s}: {scores.mean():.2%} (+/- {scores.std()*2:.2%})")
    
    print("\n" + "="*60)
    print("TRAINING AND EVALUATING ON HOLD-OUT TEST SET")
    print("="*60)
    
    results = {}
    trained_models = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        print(f"  {name} Test Accuracy: {acc:.2%}")
    
    # Create ensemble with soft voting (uses predict_proba)
    print("\n" + "="*60)
    print("TRAINING ENSEMBLE (Soft Voting)")
    print("="*60)
    
    ensemble = VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(max_iter=1000, C=1.0)),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('svm', SVC(kernel='rbf', probability=True, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
            ('mlp', MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)),
        ],
        voting='soft'  # Use probability averaging
    )
    
    # Cross-validation for ensemble
    ensemble_cv_scores = cross_val_score(ensemble, X, y, cv=cv, scoring='accuracy')
    print(f"Ensemble 5-Fold CV: {ensemble_cv_scores.mean():.2%} (+/- {ensemble_cv_scores.std()*2:.2%})")
    
    print("\nTraining ensemble classifier...")
    ensemble.fit(X_train, y_train)
    y_pred_ensemble = ensemble.predict(X_test)
    y_proba_ensemble = ensemble.predict_proba(X_test)
    ensemble_acc = accuracy_score(y_test, y_pred_ensemble)
    results['Ensemble'] = ensemble_acc
    
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY (Hold-out Test Set)")
    print(f"{'='*60}")
    for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        marker = " ★ BEST" if acc == max(results.values()) else ""
        print(f"  {name:20s}: {acc:.2%}{marker}")
    
    print(f"\n{'='*60}")
    print(f"ENSEMBLE TEST ACCURACY: {ensemble_acc:.2%}")
    print(f"{'='*60}")
    
    print("\nClassification Report (Ensemble):")
    print(classification_report(y_test, y_pred_ensemble, target_names=['Human', 'AI']))
    
    print("Confusion Matrix (Ensemble):")
    cm = confusion_matrix(y_test, y_pred_ensemble)
    print(f"  Human predicted as Human: {cm[0][0]}")
    print(f"  Human predicted as AI:    {cm[0][1]}")
    print(f"  AI predicted as Human:    {cm[1][0]}")
    print(f"  AI predicted as AI:       {cm[1][1]}")
    
    # Show misclassified samples for debugging
    print("\n" + "="*60)
    print("MISCLASSIFIED SAMPLES (Debug)")
    print("="*60)
    
    test_texts = [texts[i] for i in range(len(texts))]
    X_test_indices = []
    for i, (train_idx, test_idx) in enumerate(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X, y)):
        if i == 0:  # Use first fold for demonstration
            X_test_indices = test_idx
            break
    
    # Find misclassified in current test set
    misclassified_count = 0
    for i, (true, pred, proba) in enumerate(zip(y_test, y_pred_ensemble, y_proba_ensemble)):
        if true != pred:
            misclassified_count += 1
            label_true = "Human" if true == 0 else "AI"
            label_pred = "Human" if pred == 0 else "AI"
            confidence = proba[pred] * 100
            # Find original text (approximate - test set is shuffled)
            print(f"\n  #{misclassified_count}: True={label_true}, Predicted={label_pred} (conf: {confidence:.1f}%)")
    
    if misclassified_count == 0:
        print("  No misclassifications in test set!")
    else:
        print(f"\n  Total misclassified: {misclassified_count}/{len(y_test)}")
    
    # Show confidence distribution
    print("\n" + "="*60)
    print("CONFIDENCE ANALYSIS")
    print("="*60)
    
    low_conf = sum(1 for p in y_proba_ensemble if 0.4 <= max(p) <= 0.6)
    med_conf = sum(1 for p in y_proba_ensemble if 0.6 < max(p) <= 0.8)
    high_conf = sum(1 for p in y_proba_ensemble if max(p) > 0.8)
    
    print(f"  Low confidence (40-60%):  {low_conf} samples")
    print(f"  Medium confidence (60-80%): {med_conf} samples")
    print(f"  High confidence (>80%):   {high_conf} samples")
    
    # Retrain ensemble on full dataset for production
    print("\n" + "="*60)
    print("TRAINING PRODUCTION MODEL")
    print("="*60)
    
    ensemble_full = VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(max_iter=1000, C=1.0)),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('svm', SVC(kernel='rbf', probability=True, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
            ('mlp', MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)),
        ],
        voting='soft'
    )
    ensemble_full.fit(X, y)
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(ensemble_full, MODEL_PATH)
    
    # Also save the embedding model reference and training texts for debug mode
    debug_data = {
        'texts': texts,
        'labels': y,
        'embeddings': X,
    }
    joblib.dump(debug_data, os.path.join(MODEL_DIR, "debug_data.pkl"))
    
    print(f"\nSaved ensemble model to {MODEL_PATH}")
    print(f"Saved debug data to {os.path.join(MODEL_DIR, 'debug_data.pkl')}")

if __name__ == "__main__":
    main()
