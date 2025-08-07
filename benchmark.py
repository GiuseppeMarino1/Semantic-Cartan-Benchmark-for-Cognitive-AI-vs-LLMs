from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)

def get_llm_metrics(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = outputs.hidden_states
        embeddings = hidden_states[-1].squeeze(0).mean(dim=0).numpy()
        return embeddings, hidden_states, None

def calculate_cartan_matrix(vectors):
    scaler = StandardScaler()
    standardized_vectors = scaler.fit_transform(vectors)
    cartan_matrix = np.corrcoef(standardized_vectors)
    plt.imshow(cartan_matrix, cmap="viridis")
    plt.colorbar()
    plt.title("Matrice di Cartan Semantica")
    plt.show()
    return cartan_matrix

def semantic_entanglement(hidden_states):
    embeddings = [state.mean(dim=1).squeeze().numpy() for state in hidden_states]
    similarities = [np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)) for e1, e2 in zip(embeddings[:-1], embeddings[1:])]
    plt.plot(similarities)
    plt.title("Variazione Entanglement Semantico")
    plt.xlabel("Blocco Cognitivo")
    plt.ylabel("Similarità Coseno")
    plt.show()
    return similarities

def approximate_phi(hidden_states):
    activations = np.array([state.numpy() for state in hidden_states])
    mean_activations = activations.mean(axis=(2, 3))
    cov_matrix = np.cov(mean_activations)
    eigvals, _ = np.linalg.eigh(cov_matrix)
    phi = np.sum(eigvals / eigvals.sum())
    return phi

def run_experiment(text_blocks):
    metrics = []
    for block in text_blocks:
        embeddings, hidden_states, _ = get_llm_metrics(block)
        phi = approximate_phi(hidden_states)
        entanglement = semantic_entanglement(hidden_states)
        metrics.append((embeddings, phi, np.mean(entanglement)))
    return metrics

def simulate_qbicore_output(num_blocks):
    np.random.seed(42)
    return [(np.random.rand(1024), np.random.rand(), np.random.rand()) for _ in range(num_blocks)]

def analyze_and_visualize(llm_metrics, qbicore_metrics):
    llm_vectors = [m[0] for m in llm_metrics]
    qbicore_vectors = [m[0] for m in qbicore_metrics]

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    calculate_cartan_matrix(llm_vectors)
    plt.title("LLM Matrice Cartan")

    plt.subplot(1, 2, 2)
    calculate_cartan_matrix(qbicore_vectors)
    plt.title("QBI-Core Matrice Cartan")
    plt.show()

    llm_phi = [m[1] for m in llm_metrics]
    qbicore_phi = [m[1] for m in qbicore_metrics]

    plt.plot(llm_phi, label='LLM Φ(t)')
    plt.plot(qbicore_phi, label='QBI-Core Φ(t)')
    plt.legend()
    plt.title("Confronto Φ(t)")
    plt.xlabel("Blocco Cognitivo")
    plt.ylabel("Φ")
    plt.show()

if __name__ == "__main__":
    text_blocks = [
        "This is an example sentence.",
        "This is another example sentence for testing."
    ]
    llm_metrics = run_experiment(text_blocks)
    qbicore_metrics = simulate_qbicore_output(len(text_blocks))
    analyze_and_visualize(llm_metrics, qbicore_metrics)
