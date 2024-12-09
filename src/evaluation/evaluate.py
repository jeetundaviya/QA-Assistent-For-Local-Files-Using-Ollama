from sklearn.metrics import accuracy_score

def evaluate_qa_model(qa_chain, test_data):
    predictions = []
    for question, _ in test_data:
        response = qa_chain.run(question)
        predictions.append(response)

    # Evaluate predictions
    ground_truths = [answer for _, answer in test_data]
    accuracy = accuracy_score(ground_truths, predictions)
    return accuracy, predictions
