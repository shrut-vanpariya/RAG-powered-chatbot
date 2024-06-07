import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge

def evaluate_responses(chatbot_responses, pred_answers):
    bleu_scores = []
    rouge_scores = {'rouge-1': {'f': [], 'p': [], 'r': []}, 'rouge-2': {'f': [], 'p': [], 'r': []}, 'rouge-l': {'f': [], 'p': [], 'r': []}}

    for i in range(len(chatbot_responses)):
        chatbot_response = chatbot_responses[i]
        pred_answer = pred_answers[i]

        # Calculate BLEU score with smoothing
        smoothing_function = SmoothingFunction().method3
        bleu_score = sentence_bleu([pred_answer.split()], chatbot_response.split(), smoothing_function=smoothing_function)
        bleu_scores.append(bleu_score)

        # Calculate ROUGE score
        rouge = Rouge()
        rouge_score = rouge.get_scores(chatbot_response, pred_answer)[0]
        for metric in rouge_score.keys():
            if metric in rouge_scores:
                for key in rouge_scores[metric].keys():
                    if key in rouge_score[metric]:
                        rouge_scores[metric][key].append(rouge_score[metric][key])

    return bleu_scores, rouge_scores

def main():
    # Load data from JSON file
    with open('test_data\dataset.json', 'r') as file:
        data = json.load(file)

    questions = data['questions']
    pred_answers = [q['pred'] for q in questions]

    # Generate responses from chatbot
    chatbot_responses = []
    for q in questions:
        user_input = q['question']
        # Replace this with your chatbot's response generation code
        chatbot_response = "Sample response from chatbot"
        chatbot_responses.append(chatbot_response)

    # Evaluate responses
    bleu_scores, rouge_scores = evaluate_responses(chatbot_responses, pred_answers)

    # Print evaluation metrics
    print("BLEU Scores:", bleu_scores)
    print("ROUGE Scores:", rouge_scores)

if __name__ == "__main__":
    main()
