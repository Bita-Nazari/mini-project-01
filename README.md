Which model do you expect to perform best for fraud detection? Why?

I guess logistic regression perform better than the others bc knn model its not good choice for this dataset we have few fraud samples and i think its not gonna work good here , knn may give us good accurecy but its not gonna give us good recall
and bc our dataset is really big knn will be very lazy in this situation and its gonna calculate distance to training samples and decision tree may not work good either bc if we setmax depth big its gonna get over fit and if we set small max depth its gonna fet underfit but with norma maxdepth it may work good

Which metric is more important for this problem: Precision, Recall, or F1-score? Why?

recall,bc its important for us to find fraud transactions as possible we can

What do you expect to happen if the model predicts all transactions as legitimate?

accurecy will be hight but it doesent mesn model worksgood it get hight bc most of our samples are not fraud

Do you expect feature scaling to significantly affect KNN performance?

yes bc if we dont scale features some features value might be very big and its effect
model prediction especially in knn 

Do you expect the Decision Tree to overfit? Why?

if we get big maxdepth its gonna over fit bc it makes model complicate and its gonna work good only in train sets