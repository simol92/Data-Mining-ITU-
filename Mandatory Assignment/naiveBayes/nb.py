import numpy as np
import pandas as pd

df = pd.read_csv('data_with_clusters.csv')

# splitting data into training and testing datasets, approx 80% assigned to training data
training_data = df.sample(frac=0.8, random_state=1) 
test_data = df.drop(training_data.index)  

# dividing into training and test columns based on having target col or not. Used in the the prediction method
y_train = training_data['Target']
x_train = training_data.drop('Target', axis=1)
y_test = test_data["Target"]
x_test = test_data.drop("Target", axis=1)

# pdf function
# defines the density of the probability, that a continuous variable will lie within a particular range of values
def probabilityDensityFunction(column, mean, variance):
    st_deviation = np.sqrt(variance)
    pdf = (np.e ** (-0.5 * ((column - mean)/st_deviation) ** 2)) / (
        st_deviation * np.sqrt(2 * np.pi))
    return pdf

# calculate the mean, variance, and prior of a given dataset
def fit(sample):
    all_mean = sample.groupby(sample['Target']).mean()
    all_variance = sample.groupby(sample['Target']).var()
    prior_prob = (sample.groupby(sample['Target']).count() / len(sample)).iloc[:,1]
    classes = np.unique(sample['Target']).tolist()
    return {'means': all_mean,'variances': all_variance,
            'prior probabilities': prior_prob, 'unique classes':classes}

# Bayes' Theorem:
# P(A|B) = P(B|A) * P(A) / P(B)
# In the context of the fit function:
# P(A|B) is the posterior probability
# P(B|A) is the likelihood
# P(A) is the prior probability
# P(B) is the prior probability of event B

def predict(sample, model, xTrainData, OriginalTrainData):
    # getting all the relevant nested data from the fit object 
    all_mean = model['means']
    all_variance = model['variances']
    prior_prob = model['prior probabilities']
    classes = model['unique classes']

    predictions = [] 

    for dp in sample.index:  
        #storing the predicted values (class) that a data point can be assigned to
        predicted_values_likelihoods = []
        #getting the datapoint in the dataset  
        currentDP = sample.loc[dp] 

        for unique_value in classes:
            #storing the likelihoods for each feature / column
            colsValue_likelihoods = [] 
            #retrieve the prob of the current class from the prior probs.
            #the prior probabilities of the classes extracted from the model
            #increasing numerical stability by taking the log value of the prior probability
                #stabilizes the value 
            colsValue_likelihoods.append(np.log(prior_prob[unique_value]))  

            for column in xTrainData.columns:
                #get the datapoint value   
                dataPointValue = currentDP[column]  
                #getting the mean and variance of the feature / column for the specific class
                classMean = all_mean[column].loc[unique_value]  
                classVariance = all_variance[column].loc[unique_value]
                #calculates the pdf, which represents the likelihood of the observed features belonging to a certain class
                #if a gaussian (normal distributed) data set is present.
                #taking the log for numerical stability when dealing with small probabilities
                likelihood_of_DP = np.log(probabilityDensityFunction(dataPointValue, classMean, classVariance))  

                #store the likelihoods for each feature in the datapoint
                colsValue_likelihoods.append(likelihood_of_DP)  
            #calculate the combined likelihood for all feature of the spec. datapoint 
            combined_likelihood = sum(colsValue_likelihoods)  
            #store the combined likelihoods for all datapoints
            #overall likelihood of each DP belonging to a specific class
            predicted_values_likelihoods.append(combined_likelihood) 

        #find the highest combined likelihood, the most probable class for the features of the datapoint
        #this predicted class will get assigned to the datapoint in the prediction model
        max_index = predicted_values_likelihoods.index(max(predicted_values_likelihoods))  
        #assign the class with the best fit to the datapoint.
        prediction = classes[max_index] 
        #append it to the final list of predictions 
        predictions.append(prediction)  

    return predictions 


# Fit the model to the training data
model = fit(training_data)

# Make predictions for the training and testing datasets
predictTrain = predict(x_train, model,xTrainData=x_train,OriginalTrainData= training_data)
predictTest = predict(x_test, model, xTrainData=x_train, OriginalTrainData=training_data)

# calculating accuracy
# take the testsample and hold it up against the prediction lists.
# taking the resulting score and divides with the size and multiply with 100 = the accuracy score!
def accuracy(testSample, prediction):
    testSample = list(testSample)
    prediction = list(prediction)
    score = 0
    for i, j in zip(testSample, prediction):
        if i == j:
            score += 1
    return round((score / len(testSample) * 100), 2)

acc_train = accuracy(y_train, predictTrain)
accu_test = accuracy(y_test, predictTest)

print("Accuracy on the train data: " + str(acc_train) + "%")
print("Accuracy on the test data: " + str(accu_test) + "%")