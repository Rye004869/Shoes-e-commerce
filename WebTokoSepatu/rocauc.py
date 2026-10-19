from sklearn.metrics import roc_curve, auc

# Contoh probabilitas prediksi
y_scores = [0.1, 0.4, 0.35, 0.8, 0.65, 0.9]
y_true = [0, 0, 1, 1, 1, 1]

fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
