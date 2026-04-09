import matplotlib.pyplot as plt

def plot_model_comparison():
    models = ["Baseline", "SBERT"]
    accuracy = [0.694, 0.697]
    f1 = [0.806, 0.774]

    x = range(len(models))

    plt.figure()
    plt.bar(x, accuracy)
    plt.xticks(x, models)
    plt.title("Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.savefig("results/accuracy.png")
    plt.close()

    plt.figure()
    plt.bar(x, f1)
    plt.xticks(x, models)
    plt.title("F1 Score Comparison")
    plt.ylabel("F1 Score")
    plt.savefig("results/f1.png")
    plt.close()